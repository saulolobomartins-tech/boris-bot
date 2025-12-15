import os
import re
import io
import datetime
from datetime import date, timedelta, datetime as dt
from collections import defaultdict

from fastapi import FastAPI, Request
from pydantic import BaseModel

from dateutil.parser import parse as dateparse

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, MessageHandler,
    ContextTypes, filters
)

from supabase import create_client, Client

# -------------- OpenAI (Whisper via API) ---------------
# openai>=1.0.0
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oa_client = OpenAI(api_key=OPENAI_API_KEY)

# -------------- ENV ----------------
TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not (TOKEN and SUPABASE_URL and SUPABASE_KEY):
    raise RuntimeError("Faltam variáveis de ambiente: TELEGRAM_TOKEN, SUPABASE_URL, SUPABASE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------- FASTAPI --------------
app = FastAPI()

# =====================================================================================
#                              REGRAS & DICIONÁRIOS
# =====================================================================================

# Categorias ampliadas (regex -> nome)
CATEGORY_RULES = [
    # Materiais e disciplinas
    (r"\beletr(ic|icist|icista|ic[aá]ria|ricidade|fiação)\b", "Elétrico"),
    (r"\bhidr(aul|ául|[ao]ul)[^a-z]*|encanador|encanament", "Hidráulico"),
    (r"\bdrywall|gesso|forro\b", "Drywall/Gesso"),
    (r"\bpintur|tinta|rolo de pintura\b", "Pintura"),
    (r"\bmadeir|compensado|sarraf[oó]\b", "Madeira"),
    (r"\bferra(gens|gem)|parafus|broca|chumbador|rebite\b", "Ferragens"),
    (r"\bconcret|cimento|areia|brita|argamassa|reboco\b", "Concreto/Alvenaria"),
    (r"\bcer[aâ]mica|porcelanat|revestiment|piso vinílic|granito|mármore\b", "Revestimentos"),
    (r"\besquadri(a|a)s?|porta|janela|vidro temperado\b", "Esquadrias/Vidro"),
    (r"\bimpermeabiliza|manta asf[aá]ltica|vedacit\b", "Impermeabilização"),
    (r"\bart(e|es)e?fato de cimento|bloco estrutural\b", "Artefatos de Cimento"),

    # Equipamentos e locação
    (r"\bbobcat|retroescavadeira|munck|plataforma elevat[oó]ria|guindaste\b", "Locação de Equipamentos"),
    (r"\bcompactador|vibrador de concreto|gerador\b", "Equipamentos"),
    (r"\bferramenta|esmerilhadeira|serra circular|lixadeira\b", "Ferramentas"),

    # Logística e combustível
    (r"\bfrete|carretinha|transporte\b", "Logística"),
    (r"\bcombust[ií]vel|gasolina|etanol|diesel|posto\b", "Combustível"),

    # Mão de obra e terceiros
    (r"\bm[aã]o de obra|di[aá]ria|pedreir|ajudant|servente|aplicador\b", "Mão de Obra"),
    (r"\bprojet(o|ista)|ART|CREA|laudo|consultoria|engenheir|arquitet|top[oó]graf\b", "Projetos/Documentação"),

    # Administração e marketing
    (r"\btr[aá]fego|ads|google ads|meta|facebook|instagram\b", "Marketing"),
    (r"\baluguel|loca[cç][aã]o de sala|internet|telefone|energia do escrit[oó]rio\b", "Custos Fixos"),
    (r"\bpapelaria|impress[aã]o|cartucho|toner\b", "Insumos Administrativos"),

    # Alimentação
    (r"\bcomida|refei[cç][aã]o|lanche|marmit|almo[cç]o|jantar\b", "Alimentação"),

    # Taxas e financeiros
    (r"\btaxa|tarifa|iof|banc[aá]ria|bolet(o|os)|juros|multa\b", "Taxas/Financeiro"),
]
DEFAULT_CATEGORY = "Outros"

# Formas de pagamento (sinônimos)
PAYMENT_SYNONYMS = {
    "PIX":      r"\bpix\b|\bfiz um pix\b|\bmandei um pix\b|\bchave pix\b",
    "CREDITO":  r"\bcr[eé]dito\b|\bno cart[aã]o de cr[eé]dito\b|\bpassei no cr[eé]dito\b",
    "DEBITO":   r"\bd[eé]bito\b|\bno cart[aã]o de d[eé]bito\b|\bpassei no d[eé]bito\b",
    "DINHEIRO": r"\bdinheiro\b|\bcash\b",
    "VALE":     r"\bvale\b|\bvale(i|u)\b|\badiantamento\b",
}

# Meses PT
MONTHS_PT = {
    "janeiro":1, "fevereiro":2, "março":3, "marco":3, "abril":4, "maio":5, "junho":6,
    "julho":7, "agosto":8, "setembro":9, "outubro":10, "novembro":11, "dezembro":12
}

# Intent para consultas/relatórios
QUERY_INTENT_RE = re.compile(
    r"\b(quanto\s+(eu\s+)?gastei|gastos|relat[oó]rio|me\s+mostra|mostra\s+pra\s+mim|me\s+manda)\b",
    re.I
)

# =====================================================================================
#                               HELPERS DE PARSE
# =====================================================================================

def money_from_text(txt:str):
    s = txt.replace("R$", "").replace(" ", "")
    m = re.search(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,\d{2})?", s)
    if not m:
        return None
    raw = m.group(0).replace(".", "").replace(",", ".")
    try:
        return round(float(raw), 2)
    except:
        return None

def guess_category(txt: str):
    low = txt.lower()
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, low):
            return cat
    return DEFAULT_CATEGORY

def guess_payment(txt: str):
    low = txt.lower()
    for label, pat in PAYMENT_SYNONYMS.items():
        if re.search(pat, low):
            return label
    return None

def guess_cc(txt: str):
    """
    Detecta CC a partir de:
    'obra do X', 'reforma da Y', 'container do Z' (variações de de/da/do).
    Retorna código em caixa alta: OBRA_NOME, REFORMA_NOME, CONTAINER_NOME
    """
    low = txt.lower()
    m = re.search(r"\b(obra|reforma|container)\s+(do|da|de)\s+([a-zà-ú\s]+)", low, re.I)
    if m:
        base = m.group(1).upper()        # OBRA / REFORMA / CONTAINER
        nome = m.group(3).strip()
        nome = re.sub(r"[^a-zà-ú\s]", "", nome, flags=re.I)
        nome = " ".join(nome.split()[:3])
        nome_code = re.sub(r"\s+", "_", nome).upper()
        return f"{base}_{nome_code}"
    return None

def get_or_none(res):
    return res.data if hasattr(res, "data") else res

def _first_day_of_week(d: date):
    return d - timedelta(days=d.weekday())

def _last_day_of_week(d: date):
    return _first_day_of_week(d) + timedelta(days=7)

def parse_period_pt(text: str):
    """
    Retorna (start_date_iso, end_date_iso_exclusive, label)
    """
    low = text.lower().strip()
    today = date.today()

    if re.search(r"\bhoje\b", low):
        return today.isoformat(), (today + timedelta(days=1)).isoformat(), "hoje"

    if re.search(r"\bontem\b", low):
        y = today - timedelta(days=1)
        return y.isoformat(), today.isoformat(), "ontem"

    if re.search(r"\bessa semana\b", low):
        s = _first_day_of_week(today)
        e = _last_day_of_week(today)
        return s.isoformat(), e.isoformat(), "essa semana"

    if re.search(r"\bsemana passada\b", low):
        e = _first_day_of_week(today)
        s = e - timedelta(days=7)
        return s.isoformat(), e.isoformat(), "semana passada"

    if re.search(r"\besse m[eê]s\b", low):
        s = today.replace(day=1)
        e = (s.replace(day=28) + timedelta(days=4)).replace(day=1)
        return s.isoformat(), e.isoformat(), "este mês"

    if re.search(r"\bm[eê]s passado\b", low):
        s_atual = today.replace(day=1)
        e_passado = s_atual
        s_passado = (s_atual - timedelta(days=1)).replace(day=1)
        return s_passado.isoformat(), e_passado.isoformat(), "mês passado"

    if re.search(r"\besse ano\b", low):
        s = date(today.year, 1, 1)
        e = date(today.year + 1, 1, 1)
        return s.isoformat(), e.isoformat(), "este ano"

    if re.search(r"\bano passado\b", low):
        s = date(today.year - 1, 1, 1)
        e = date(today.year, 1, 1)
        return s.isoformat(), e.isoformat(), "ano passado"

    m = re.search(r"\bem\s+(janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)(?:\s+de\s+(\d{4}))?", low)
    if not m:
        m = re.search(r"\b(janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)(?:\s+de\s+(\d{4}))\b", low)
    if m:
        mes = m.group(1).replace("ç","c")
        ano = int(m.group(2)) if m.group(2) else today.year
        month_num = MONTHS_PT.get(mes, None)
        if month_num:
            s = date(ano, month_num, 1)
            e = (s.replace(day=28) + timedelta(days=4)).replace(day=1)
            return s.isoformat(), e.isoformat(), f"{m.group(1).title()} {ano}"

    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\s*(?:a|até)\s*(\d{1,2})/(\d{1,2})(?:/(\d{4}))?", low)
    if m:
        d1, m1, y1 = int(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else today.year
        d2, m2, y2 = int(m.group(4)), int(m.group(5)), int(m.group(6)) if m.group(6) else y1
        s = date(y1, m1, d1)
        e = date(y2, m2, d2) + timedelta(days=1)
        return s.isoformat(), e.isoformat(), f"{d1:02d}/{m1:02d}/{y1} a {d2:02d}/{m2:02d}/{y2}"

    s = today.replace(day=1)
    e = (s.replace(day=28) + timedelta(days=4)).replace(day=1)
    return s.isoformat(), e.isoformat(), "este mês (padrão)"

def guess_category_filter(text: str):
    low = text.lower()
    for pat, name in CATEGORY_RULES:
        if re.search(pat, low):
            return name
    return None

def guess_paid_filter(text: str):
    low = text.lower()
    for label, pat in PAYMENT_SYNONYMS.items():
        if re.search(pat, low):
            return label
    return None

def guess_cc_filter(text: str):
    return guess_cc(text)

def is_income_query(text: str):
    return bool(re.search(r"\b(entrou|recebi|receitas?|quanto\s+entrou)\b", text, re.I))

def is_report_intent(text: str):
    return bool(QUERY_INTENT_RE.search(text))

# =====================================================================================
#                               PERSISTÊNCIA
# =====================================================================================

def save_entry(tg_user_id:int, txt:str):
    amount = money_from_text(txt)
    if amount is None:
        return False, "Não achei o valor. Ex.: 'paguei 200 no eletricista'."

    etype = "income" if re.search(r"\b(recebi|receita|entrada|vendi|entrou)\b", txt, re.I) else "expense"

    cat_name = guess_category(txt)
    cc_code  = guess_cc(txt)
    paid_via = guess_payment(txt)

    u = sb.table("users").select("*").eq("tg_user_id", tg_user_id).execute()
    ud = get_or_none(u)
    if not ud or not ud[0]["is_active"]:
        return False, "Usuário não autorizado. Use /start e peça autorização."
    user_id = ud[0]["id"]
    role = ud[0]["role"]

    c = sb.table("categories").select("id").eq("name", cat_name).execute()
    cd = get_or_none(c)
    cat_id = cd[0]["id"] if cd else None

    cc_id = None
    if cc_code:
        cc = sb.table("cost_centers").select("id").eq("code", cc_code).execute()
        ccd = get_or_none(cc)
        if ccd:
            cc_id = ccd[0]["id"]

    status = "approved" if role in ("owner","partner") else "pending"

    sb.table("entries").insert({
        "entry_date": datetime.date.today().isoformat(),
        "type": etype,
        "amount": amount,
        "description": txt,
        "category_id": cat_id,
        "cost_center_id": cc_id,
        "paid_via": paid_via,
        "created_by": user_id,
        "status": status
    }).execute()

    return True, {"amount": amount, "type": etype, "category": cat_name, "cc": cc_code, "status": status, "paid_via": paid_via}

# =====================================================================================
#                               CONSULTAS / RELATÓRIOS
# =====================================================================================

async def run_query_and_reply(update: Update, text: str):
    start, end, label = parse_period_pt(text)
    cat = guess_category_filter(text)
    paid = guess_paid_filter(text)
    cc_code = guess_cc_filter(text)
    is_income = is_income_query(text)

    q = sb.table("entries").select("amount,category_id,cost_center_id,paid_via,type,entry_date")\
        .gte("entry_date", start).lt("entry_date", end)

    q = q.eq("type", "income" if is_income else "expense")

    if paid:
        q = q.eq("paid_via", paid)

    if cat:
        c = sb.table("categories").select("id").eq("name", cat).execute()
        cd = get_or_none(c)
        if cd:
            q = q.eq("category_id", cd[0]["id"])

    if cc_code:
        cc = sb.table("cost_centers").select("id").eq("code", cc_code).execute()
        ccd = get_or_none(cc)
        if ccd:
            q = q.eq("cost_center_id", ccd[0]["id"])

    rows = get_or_none(q.execute()) or []
    total = sum(float(r["amount"]) for r in rows)
    moeda = lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    escopo = "receitas" if is_income else "gastos"
    filtros = []
    if cat: filtros.append(cat)
    if paid: filtros.append(paid.title())
    if cc_code: filtros.append(cc_code)
    filtros_txt = f" | Filtros: {', '.join(filtros)}" if filtros else ""

    await update.message.reply_text(
        f"📊 Total de {escopo} em {label}{filtros_txt}:\n*{moeda(total)}*",
        parse_mode="Markdown"
    )

# =====================================================================================
#                               TELEGRAM HANDLERS
# =====================================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    exist = sb.table("users").select("*").eq("tg_user_id", u.id).execute()
    data = get_or_none(exist)
    if not data:
        sb.table("users").insert({
            "tg_user_id": u.id, "name": u.full_name, "role": "viewer", "is_active": False
        }).execute()

    await update.message.reply_text(
        f"Fala, {u.first_name}! Eu sou o Boris.\n"
        f"Teu Telegram user id é: {u.id}\n"
        f"Pede pro owner te autorizar com /autorizar {u.id} role=buyer"
    )

async def cmd_autorizar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    q = sb.table("users").select("role,is_active").eq("tg_user_id", u.id).execute()
    you = get_or_none(q)
    if not you or you[0]["role"] != "owner" or not you[0]["is_active"]:
        await update.message.reply_text("Somente o owner pode autorizar usuários.")
        return

    if len(context.args) == 0:
        await update.message.reply_text("Uso: /autorizar <tg_user_id> role=owner|partner|buyer|viewer")
        return
    target = int(context.args[0])
    role = "buyer"
    for a in context.args[1:]:
        if a.startswith("role="):
            role = a.split("=",1)[1]

    sb.table("users").upsert({"tg_user_id": target, "role": role, "is_active": True, "name": ""}).execute()
    await update.message.reply_text(f"Usuário {target} autorizado como {role} ✅")

async def cmd_despesa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args) if context.args else (update.message.text or "")
    ok, res = save_entry(update.effective_user.id, txt)
    if ok:
        r = res
        paid_str = f" • {r['paid_via']}" if r.get("paid_via") else ""
        await update.message.reply_text(
            f"✅ Lançado: R$ {r['amount']:.2f} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{paid_str}"
        )
    else:
        await update.message.reply_text(f"⚠️ {res}")

async def cmd_receita(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args) if context.args else (update.message.text or "")
    ok, res = save_entry(update.effective_user.id, "receita "+txt)
    if ok:
        r = res
        await update.message.reply_text(
            f"✅ Receita: R$ {r['amount']:.2f} • {r['category']} • {r['cc'] or 'Sem CC'}"
        )
    else:
        await update.message.reply_text(f"⚠️ {res}")

async def cmd_relatorio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Mantém um resumo do mês por categoria/CC (exemplo simples)
    today = datetime.date.today()
    month_start = today.replace(day=1).isoformat()
    month_end = (today.replace(day=28) + timedelta(days=4)).replace(day=1).isoformat()

    resp = sb.table("entries").select("amount,category_id,cost_center_id,type,entry_date,status")\
        .gte("entry_date", month_start).lt("entry_date", month_end).eq("type","expense").execute()
    rows = get_or_none(resp) or []

    cats = {r["id"]: r["name"] for r in get_or_none(sb.table("categories").select("id,name").execute())}
    ccs  = {r["id"]: r["code"] for r in get_or_none(sb.table("cost_centers").select("id,code").execute())}

    by_cat = defaultdict(float)
    by_cc  = defaultdict(float)
    total = 0.0
    for r in rows:
        total += float(r["amount"])
        by_cat[cats.get(r["category_id"],"Sem categoria")] += float(r["amount"])
        by_cc[ccs.get(r["cost_center_id"],"Sem CC")] += float(r["amount"])

    def fmt(d):
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"• {k}: R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") for k,v in items]) or "• (sem lançamentos)"

    moeda = lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    msg = (
        f"📊 *Resumo do mês*\n"
        f"Total: *{moeda(total)}*\n\n"
        f"Por categoria:\n{fmt(by_cat)}\n\n"
        f"Por centro de custo:\n{fmt(by_cc)}"
    )
    await update.message.reply_markdown(msg)

# -------------------- TEXTO --------------------
async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or "").strip()

    # Consulta/relatório por texto
    if is_report_intent(user_text):
        await run_query_and_reply(update, user_text)
        return

    # Lançamento normal
    ok, res = save_entry(update.effective_user.id, user_text)
    if ok:
        r = res
        paid_str = f" • {r['paid_via']}" if r.get("paid_via") else ""
        await update.message.reply_text(
            f"✅ Lançado: R$ {r['amount']:.2f} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{paid_str}"
        )
    else:
        await update.message.reply_text(
            "Me manda algo tipo: 'paguei 200 no eletricista da obra do Rodrigo (pix)'\n"
            "ou usa /despesa 1200 tráfego pago SEDE"
        )

# -------------------- ÁUDIO (voice/audio) --------------------
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not OPENAI_API_KEY:
        await update.message.reply_text("Whisper não está configurado (OPENAI_API_KEY ausente).")
        return

    file = None
    if update.message.voice:
        file = await update.message.voice.get_file()
        filename = "audio.ogg"
    elif update.message.audio:
        file = await update.message.audio.get_file()
        # tenta manter a extensão
        filename = update.message.audio.file_name or "audio.mp3"
    else:
        return

    bio = io.BytesIO()
    await file.download(out=bio)
    bio.seek(0)

    try:
        # Whisper via OpenAI API (models: "gpt-4o-mini-transcribe" ou "whisper-1")
        transcript = oa_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=(filename, bio)
        )
        transcrito = transcript.text.strip()
    except Exception as e:
        await update.message.reply_text(
            f"Não consegui transcrever o áudio. Erro: {e}"
        )
        return

    # Se for consulta, responde e sai
    if is_report_intent(transcrito):
        await update.message.reply_text(f"🗣️ Transcrito: “{transcrito}”")
        await run_query_and_reply(update, transcrito)
        return

    # Lançamento padrão
    ok, res = save_entry(update.effective_user.id, transcrito)
    if ok:
        r = res
        paid_str = f" • {r['paid_via']}" if r.get("paid_via") else ""
        await update.message.reply_text(
            f"🗣️ Transcrito: “{transcrito}”\n"
            f"✅ Lançado: R$ {r['amount']:.2f} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{paid_str}"
        )
    else:
        await update.message.reply_text(
            f"🗣️ Transcrito: “{transcrito}”\n"
            f"⚠️ {res}"
        )

# =====================================================================================
#                               TELEGRAM APP
# =====================================================================================

tg_app: Application = ApplicationBuilder().token(TOKEN).build()

# Comandos
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("autorizar", cmd_autorizar))
tg_app.add_handler(CommandHandler("despesa", cmd_despesa))
tg_app.add_handler(CommandHandler("receita", cmd_receita))
tg_app.add_handler(CommandHandler("relatorio", cmd_relatorio))

# Mensagens
tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))

# Inicialização/encerramento obrigatórios p/ webhook (ptb v20)
@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    await tg_app.start()

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

# =====================================================================================
#                               FASTAPI ENDPOINTS
# =====================================================================================

class TgUpdate(BaseModel):
    update_id: int | None = None

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

@app.get("/")
def alive():
    return {"boris": "ok"}
