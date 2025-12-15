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

CATEGORY_RULES = [
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
    (r"\bbobcat|retroescavadeira|munck|plataforma elevat[oó]ria|guindaste\b", "Locação de Equipamentos"),
    (r"\bcompactador|vibrador de concreto|gerador\b", "Equipamentos"),
    (r"\bferramenta|esmerilhadeira|serra circular|lixadeira\b", "Ferramentas"),
    (r"\bfrete|carretinha|transporte\b", "Logística"),
    (r"\bcombust[ií]vel|gasolina|etanol|diesel|posto\b", "Combustível"),
    (r"\bm[aã]o de obra|di[aá]ria|pedreir|ajudant|servente|aplicador\b", "Mão de Obra"),
    (r"\bprojet(o|ista)|ART|CREA|laudo|consultoria|engenheir|arquitet|top[oó]graf\b", "Projetos/Documentação"),
    (r"\btr[aá]fego|ads|google ads|meta|facebook|instagram\b", "Marketing"),
    (r"\baluguel|loca[cç][aã]o de sala|internet|telefone|energia do escrit[oó]rio\b", "Custos Fixos"),
    (r"\bpapelaria|impress[aã]o|cartucho|toner\b", "Insumos Administrativos"),
    (r"\bcomida|refei[cç][aã]o|lanche|marmit|almo[cç]o|jantar\b", "Alimentação"),
    (r"\btaxa|tarifa|iof|banc[aá]ria|bolet(o|os)|juros|multa\b", "Taxas/Financeiro"),
]
DEFAULT_CATEGORY = "Outros"

PAYMENT_SYNONYMS = {
    "PIX":      r"\bpix\b|\bfiz um pix\b|\bmandei um pix\b|\bchave pix\b",
    "CREDITO":  r"\bcr[eé]dito\b|\bno cart[aã]o de cr[eé]dito\b|\bpassei no cr[eé]dito\b",
    "DEBITO":   r"\bd[eé]bito\b|\bno cart[aã]o de d[eé]bito\b|\bpassei no d[eé]bito\b",
    "DINHEIRO": r"\bdinheiro\b|\bcash\b",
    "VALE":     r"\bvale\b|\bvale(i|u)\b|\badiantamento\b",
}

MONTHS_PT = {
    "janeiro":1, "fevereiro":2, "março":3, "marco":3, "abril":4, "maio":5, "junho":6,
    "julho":7, "agosto":8, "setembro":9, "outubro":10, "novembro":11, "dezembro":12
}

QUERY_INTENT_RE = re.compile(
    r"\b(quanto\s+(eu\s+)?gastei|gastos|relat[oó]rio|me\s+mostra|mostra\s+pra\s+mim|me\s+manda)\b",
    re.I
)

# =====================================================================================
#                               HELPERS
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
    low = txt.lower()
    m = re.search(r"\b(obra|reforma|container)\s+(do|da|de)\s+([a-zà-ú\s]+)", low, re.I)
    if m:
        base = m.group(1).upper()
        nome = m.group(3).strip()
        nome = re.sub(r"[^a-zà-ú\s]", "", nome, flags=re.I)
        nome = " ".join(nome.split()[:3])
        nome_code = re.sub(r"\s+", "_", nome).upper()
        return f"{base}_{nome_code}"
    return None

def get_or_none(res):
    return res.data if hasattr(res, "data") else res

def _first_day_of_week(d: date): return d - timedelta(days=d.weekday())
def _last_day_of_week(d: date): return _first_day_of_week(d) + timedelta(days=7)

def parse_period_pt(text: str):
    low = text.lower().strip()
    today = date.today()
    if re.search(r"\bhoje\b", low): return today.isoformat(), (today+timedelta(days=1)).isoformat(), "hoje"
    if re.search(r"\bessa semana\b", low): s=_first_day_of_week(today); e=_last_day_of_week(today); return s.isoformat(), e.isoformat(),"essa semana"
    if re.search(r"\besse m[eê]s\b", low): s=today.replace(day=1); e=(s.replace(day=28)+timedelta(days=4)).replace(day=1); return s.isoformat(),e.isoformat(),"este mês"
    if re.search(r"\bsemana passada\b", low): e=_first_day_of_week(today); s=e-timedelta(days=7); return s.isoformat(),e.isoformat(),"semana passada"
    if re.search(r"\bm[eê]s passado\b", low): s_atual=today.replace(day=1); e_passado=s_atual; s_passado=(s_atual-timedelta(days=1)).replace(day=1); return s_passado.isoformat(),e_passado.isoformat(),"mês passado"
    return today.replace(day=1).isoformat(), (today.replace(day=28)+timedelta(days=4)).replace(day=1).isoformat(), "este mês (padrão)"

def is_income_query(text: str): return bool(re.search(r"\b(entrou|recebi|receitas?)\b", text, re.I))
def is_report_intent(text: str): return bool(QUERY_INTENT_RE.search(text))

# =====================================================================================
#                               PERSISTÊNCIA
# =====================================================================================

def save_entry(tg_user_id:int, txt:str):
    amount = money_from_text(txt)
    if amount is None: return False, "Não achei o valor. Ex.: 'paguei 200 no eletricista'."
    etype = "income" if re.search(r"\b(recebi|receita|entrada|vendi|entrou)\b", txt, re.I) else "expense"
    cat_name, cc_code, paid_via = guess_category(txt), guess_cc(txt), guess_payment(txt)

    u = sb.table("users").select("*").eq("tg_user_id", tg_user_id).execute()
    ud = get_or_none(u)
    if not ud or not ud[0]["is_active"]: return False, "Usuário não autorizado. Use /start e peça autorização."
    user_id, role = ud[0]["id"], ud[0]["role"]

    c = sb.table("categories").select("id").eq("name", cat_name).execute()
    cd = get_or_none(c)
    cat_id = cd[0]["id"] if cd else None

    cc_id = None
    if cc_code:
        cc = sb.table("cost_centers").select("id").eq("code", cc_code).execute()
        ccd = get_or_none(cc)
        if ccd: cc_id = ccd[0]["id"]

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
    cat, paid, cc_code = guess_category(text), guess_payment(text), guess_cc(text)
    is_income = is_income_query(text)
    q = sb.table("entries").select("amount,category_id,cost_center_id,paid_via,type,entry_date").gte("entry_date", start).lt("entry_date", end)
    q = q.eq("type", "income" if is_income else "expense")
    if paid: q = q.eq("paid_via", paid)
    if cat:
        c = sb.table("categories").select("id").eq("name", cat).execute()
        cd = get_or_none(c)
        if cd: q = q.eq("category_id", cd[0]["id"])
    if cc_code:
        cc = sb.table("cost_centers").select("id").eq("code", cc_code).execute()
        ccd = get_or_none(cc)
        if ccd: q = q.eq("cost_center_id", ccd[0]["id"])
    rows = get_or_none(q.execute()) or []
    total = sum(float(r["amount"]) for r in rows)
    moeda = lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    escopo = "receitas" if is_income else "gastos"
    await update.message.reply_text(f"📊 Total de {escopo} em {label}: *{moeda(total)}*", parse_mode="Markdown")

# =====================================================================================
#                               TELEGRAM HANDLERS
# =====================================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    exist = sb.table("users").select("*").eq("tg_user_id", u.id).execute()
    data = get_or_none(exist)
    if not data:
        sb.table("users").insert({"tg_user_id": u.id, "name": u.full_name, "role": "viewer", "is_active": False}).execute()
    await update.message.reply_text(f"Fala, {u.first_name}! Eu sou o Boris.\nTeu Telegram user id é: {u.id}\nPede pro owner te autorizar com /autorizar {u.id} role=buyer")

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
        filename = update.message.audio.file_name or "audio.mp3"
    else:
        return

    bio = io.BytesIO()
    await file.download(out=bio)
    bio.seek(0)
    bio.name = filename

    try:
        transcript = oa_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=bio
        )
        transcrito = transcript.text.strip()
    except Exception:
        bio.seek(0)
        transcript = oa_client.audio.transcriptions.create(
            model="whisper-1",
            file=bio
        )
        transcrito = transcript.text.strip()

    if is_report_intent(transcrito):
        await update.message.reply_text(f"🗣️ Transcrito: “{transcrito}”")
        await run_query_and_reply(update, transcrito)
        return

    ok, res = save_entry(update.effective_user.id, transcrito)
    if ok:
        r = res
        paid_str = f" • {r['paid_via']}" if r.get("paid_via") else ""
        await update.message.reply_text(
            f"🗣️ Transcrito: “{transcrito}”\n✅ Lançado: R$ {r['amount']:.2f} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{paid_str}"
        )
    else:
        await update.message.reply_text(f"🗣️ Transcrito: “{transcrito}”\n⚠️ {res}")

# =====================================================================================
#                               TELEGRAM APP
# =====================================================================================

tg_app: Application = ApplicationBuilder().token(TOKEN).build()
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    await tg_app.start()

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

class TgUpdate(BaseModel):
    update_id: int | None = None

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

@app.get("/")
def alive(): return {"boris": "ok"}
