import os, re, datetime, asyncio, uuid, unicodedata
from collections import defaultdict
from fastapi import FastAPI, Request
from pydantic import BaseModel
from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    MessageHandler, ContextTypes, filters
)
from supabase import create_client, Client

# ============== CONFIG ==============
TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # para Whisper (áudio)

if not (TOKEN and SUPABASE_URL and SUPABASE_KEY):
    raise RuntimeError("Faltam variáveis de ambiente: TELEGRAM_TOKEN, SUPABASE_URL, SUPABASE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI client (Whisper)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# FastAPI
app = FastAPI()


# ============== HELPERS ==============
def _norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def moeda_fmt(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_or_none(res):
    return res.data if hasattr(res, "data") else res


# ---------- Datas naturais ----------
def parse_date_pt(txt: str) -> str | None:
    """
    Entende datas e devolve YYYY-MM-DD:
      - hoje, ontem, anteontem, amanhã
      - 'dia 12', '12/10/2025', '12-10', '12-10-25'
      - dias da semana: segunda..domingo (última ocorrência)
      - 'semana passada', 'mês passado'
    """
    t = _norm(txt)
    today = datetime.date.today()

    if "hoje" in t:
        return today.isoformat()
    if "ontem" in t:
        return (today - datetime.timedelta(days=1)).isoformat()
    if "anteontem" in t:
        return (today - datetime.timedelta(days=2)).isoformat()
    if "amanha" in t:
        return (today + datetime.timedelta(days=1)).isoformat()
    if "semana passada" in t:
        wd = today.weekday()  # 0=segunda
        last_monday = today - datetime.timedelta(days=wd+7)
        return last_monday.isoformat()
    if "mes passado" in t or "mês passado" in txt:
        first = today.replace(day=1)
        last_month = (first - datetime.timedelta(days=1)).replace(day=1)
        return last_month.isoformat()

    week_map = {
        "segunda": 0, "terca": 1, "terça": 1, "quarta": 2,
        "quinta": 3, "sexta": 4, "sabado": 5, "sábado": 5, "domingo": 6
    }
    for k, wd_target in week_map.items():
        if k in t:
            wd_today = today.weekday()
            delta = (wd_today - wd_target) % 7
            return (today - datetime.timedelta(days=delta)).isoformat()

    # dd/mm(/yy|yyyy) ou dd-mm(-yy|yyyy)
    m = re.search(r"\b(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{2,4}))?\b", t)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        d, mo = int(d), int(mo)
        if y:
            y = int(y)
            if y < 100:
                y += 2000
        else:
            y = today.year
        try:
            return datetime.date(y, mo, d).isoformat()
        except ValueError:
            pass

    m2 = re.search(r"\bdia\s+(\d{1,2})\b", t)
    if m2:
        d = int(m2.group(1))
        try:
            return datetime.date(today.year, today.month, d).isoformat()
        except ValueError:
            pass

    return None


# ---------- Valor ----------
def money_from_text(txt: str):
    s = _norm(txt).replace("r$", "").replace(" ", "")
    m = re.search(r"(-?\d{1,3}(?:\.\d{3})+|-?\d+)(?:,\d{2})?", s)
    if not m:
        return None
    raw = m.group(0).replace(".", "").replace(",", ".")
    try:
        return round(float(raw), 2)
    except:
        return None


# ---------- Tipo (despesa/receita) ----------
def guess_type(txt: str):
    t = _norm(txt)
    if re.search(r"\b(recebi|receita|entrada|vendi|entrou|aluguel recebido)\b", t):
        return "income"
    return "expense"


# ---------- Forma de pagamento ----------
def guess_payment_method(txt: str) -> str | None:
    t = _norm(txt)
    # PIX
    if re.search(r"\bpix\b|mandei um pix|fiz pix|pix p|pix pro|pix pra", t):
        return "PIX"
    # CRÉDITO
    if re.search(r"credito|cr[eé]dito|cartao de credito|passei no credito|passei no cartao", t):
        return "CREDITO"
    # DÉBITO
    if re.search(r"debito|d[eé]bito|cartao de debito|passei no debito", t):
        return "DEBITO"
    # DINHEIRO
    if re.search(r"dinheiro|em especie|em espécie|cash", t):
        return "DINHEIRO"
    # VALE
    if re.search(r"\bvale\b|valeu? (p|pro|pra)|vale para", t):
        return "VALE"
    return None


# ---------- Categorias ----------
CATEGORY_RULES = [
    # Mão de obra e serviços
    (r"\bma(o|ao)\s*de\s*obra|diaria|diaria(s)?|pedreir|ajudant|servente|marceneir|soldador", "Mão de Obra"),
    (r"eletricist|eletric|fio|disjuntor|quadro|tomada|interruptor|spot|led", "Elétrico"),
    (r"hidraul|hidrauli|cano|tubo pex|regist|torneira|ralo|caixa d'?agua|esgoto|bomba", "Hidráulico"),
    (r"drywall|forro|gesso|placa acartonad", "Drywall"),
    (r"pintur|tinta|massa corrida|lixa|rolo|fita crepe", "Pintura"),
    (r"cimento|areia|brita|argamassa|reboco|concreto|graute|bloco ceram|vergalh|armacao|forma", "Estrutura / Alvenaria"),
    (r"telha|calha|ruf|cumeeira|aluminio|zinco", "Cobertura"),
    (r"ferra|parafus|broca|eletrodo|disco corte|abracadeira|abraçadeira", "Ferragens"),
    (r"porta|janela|vidro|esquadria|fechadur|dobradic|dobradiça", "Esquadrias/Vidro"),
    (r"granito|porcelanato|piso|rodape|rodap[eé]|revestimento|argamassacol|rejunte", "Acabamento"),
    # Logística e equipamentos
    (r"uber|frete|entrega|logistic|combust|diesel|gasolina|oleo|óleo|lubrificante", "Logística"),
    (r"bobcat|compactador|gerador|betoneira|aluguel equip|locacao equip|locação equip", "Equipamentos"),
    # Adm/marketing/financeiro
    (r"trafego|tr[aá]fego|ads|google|meta|facebook|instagram|impulsionamento|anuncio|anúncio", "Marketing"),
    (r"aluguel|locacao|locação|internet|energia|conta de luz|conta de agua|água|telefone|contabilidade", "Custos Fixos"),
    (r"taxa|emolumento|cartorio|cartório|crea|art|multa|juros|tarifa|banco|ted\b|pix\b", "Taxas/Financeiro"),
]

def guess_category(txt: str):
    low = _norm(txt)
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, low):
            return cat
    return "Outros"


# ---------- Centro de custo (obra/reforma/container do/da/de + nome) ----------
def _slugify_name(name: str) -> str:
    # deixa nome em formato consistente p/ code
    s = _norm(name)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s.upper()

def guess_cc_from_project(txt: str) -> str | None:
    """
    Reconhece: 'obra do rodrigo', 'reforma rodrigo', 'container da joana',
    'obra de castanhal', 'reforma do prédio tal', etc.
    Retorna um code padronizado, ex.: 'OBRA_RODRIGO' / 'REFORMA_RODRIGO' / 'CONTAINER_JOANA'
    """
    t = _norm(txt)

    # padrões: (obra|reforma|container) [do|da|de]? <nome livre>
    m = re.search(r"\b(obra|reforma|container)\s+(?:do|da|de)?\s+([a-z0-9][a-z0-9\s\-_.]+)\b", t)
    if m:
        tipo = m.group(1)  # obra|reforma|container
        nome = m.group(2).strip()
        # corta se aparecer outra palavra "ruidosa" comum após o nome
        nome = re.split(r"\b(por|pra|pro|no|na|em|para|paguei|gastei|comprei|recebi|pix|credito|cr[eé]dito|debito|d[eé]bito)\b", nome)[0].strip()
        if nome:
            return f"{tipo.upper()}_{_slugify_name(nome)}"

    # variações reduzidas: 'obra rodrigo', 'reforma joana'
    m2 = re.search(r"\b(obra|reforma|container)\s+([a-z0-9][a-z0-9\s\-_.]+)\b", t)
    if m2:
        tipo = m2.group(1)
        nome = m2.group(2).strip()
        nome = re.split(r"\b(por|pra|pro|no|na|em|para|paguei|gastei|comprei|recebi|pix|credito|cr[eé]dito|debito|d[eé]bito)\b", nome)[0].strip()
        if nome:
            return f"{tipo.upper()}_{_slugify_name(nome)}"

    return None


# ============== CORE DE LANÇAMENTO ==============
def _ensure_cost_center(code: str) -> int | None:
    """
    Garante que exista um centro de custo com 'code'. Se não existir, cria.
    Retorna id ou None.
    """
    try:
        res = sb.table("cost_centers").select("id").eq("code", code).execute()
        rows = get_or_none(res) or []
        if rows:
            return rows[0]["id"]
        # tenta criar
        ins = sb.table("cost_centers").insert({"code": code, "name": code}).execute()
        created = get_or_none(ins) or []
        if created:
            return created[0]["id"]
        # se a API do Supabase devolver vazio, tenta buscar de novo
        res2 = sb.table("cost_centers").select("id").eq("code", code).execute()
        rows2 = get_or_none(res2) or []
        return rows2[0]["id"] if rows2 else None
    except Exception:
        return None

def save_entry(tg_user_id: int, txt: str):
    amount = money_from_text(txt)
    if amount is None:
        return False, "Não achei o valor. Ex.: 'paguei 200 no eletricista'."

    etype = guess_type(txt)
    cat_name = guess_category(txt)
    cc_code = guess_cc_from_project(txt)  # <<< projeto por obra/reforma/container ...
    dtx = parse_date_pt(txt)
    entry_date = dtx or datetime.date.today().isoformat()
    pay_method = guess_payment_method(txt)

    # usuário
    u = sb.table("users").select("*").eq("tg_user_id", tg_user_id).execute()
    ud = get_or_none(u)
    if not ud or not ud[0]["is_active"]:
        return False, "Usuário não autorizado. Use /start e peça autorização."
    user_id = ud[0]["id"]
    role = ud[0]["role"]

    # category id
    c = sb.table("categories").select("id").eq("name", cat_name).execute()
    cd = get_or_none(c)
    cat_id = cd[0]["id"] if cd else None

    # cost center id (cria se não existir)
    cc_id = None
    if cc_code:
        maybe_id = _ensure_cost_center(cc_code)
        if maybe_id:
            cc_id = maybe_id

    status = "approved" if role in ("owner", "partner") else "pending"

    # description enriquecida com método de pagamento (se não existir coluna específica)
    description = txt
    if pay_method:
        tag = f"[pagamento: {pay_method}]"
        if tag.lower() not in _norm(description):
            description = f"{txt.strip()} {tag}"

    # tenta inserir com coluna payment_method (se existir). Se falhar, cai sem ela.
    base_payload = {
        "entry_date": entry_date,
        "type": etype,
        "amount": amount,
        "description": description,
        "category_id": cat_id,
        "cost_center_id": cc_id,
        "created_by": user_id,
        "status": status
    }

    try:
        # tenta com payment_method (caso a coluna exista)
        payload = dict(base_payload)
        if pay_method:
            payload["payment_method"] = pay_method  # só funciona se a coluna existir
        sb.table("entries").insert(payload).execute()
    except Exception:
        # insere sem a coluna extra
        sb.table("entries").insert(base_payload).execute()

    return True, {
        "amount": amount,
        "type": etype,
        "category": cat_name,
        "cc": cc_code,
        "status": status,
        "entry_date": entry_date,
        "payment_method": pay_method
    }


# ============== COMANDOS ==============
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
            role = a.split("=", 1)[1]

    sb.table("users").upsert({"tg_user_id": target, "role": role, "is_active": True, "name": ""}).execute()
    await update.message.reply_text(f"Usuário {target} autorizado como {role} ✅")

async def cmd_despesa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args) if context.args else (update.message.text or "")
    ok, res = save_entry(update.effective_user.id, txt)
    if ok:
        r = res
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("payment_method"): extras.append(f"💳 {r['payment_method']}")
        tail = ("\n" + " • ".join(extras)) if extras else ""
        await update.message.reply_text(
            f"✅ Lançado: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{tail}"
        )
    else:
        await update.message.reply_text(f"⚠️ {res}")

async def cmd_receita(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args) if context.args else (update.message.text or "")
    ok, res = save_entry(update.effective_user.id, "receita " + txt)
    if ok:
        r = res
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("payment_method"): extras.append(f"💳 {r['payment_method']}")
        tail = ("\n" + " • ".join(extras)) if extras else ""
        await update.message.reply_text(
            f"✅ Receita: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'}{tail}"
        )
    else:
        await update.message.reply_text(f"⚠️ {res}")

async def cmd_relatorio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = datetime.date.today()
    month_start = today.replace(day=1).isoformat()
    month_end = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1).isoformat()

    resp = sb.table("entries").select(
        "amount,category_id,cost_center_id,type,entry_date,status"
    ).gte("entry_date", month_start).lt("entry_date", month_end).eq("type", "expense").execute()
    rows = get_or_none(resp) or []

    cats = {r["id"]: r["name"] for r in (get_or_none(sb.table("categories").select("id,name").execute()) or [])}
    ccs  = {r["id"]: r["code"] for r in (get_or_none(sb.table("cost_centers").select("id,code").execute()) or [])}

    by_cat = defaultdict(float)
    by_cc  = defaultdict(float)
    total = 0.0
    for r in rows:
        v = float(r["amount"])
        total += v
        by_cat[cats.get(r["category_id"], "Sem categoria")] += v
        by_cc[ccs.get(r["cost_center_id"], "Sem CC")] += v

    def fmt(d):
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"• {k}: {moeda_fmt(v)}" for k, v in items]) or "• (sem lançamentos)"

    msg = (
        f"📊 *Resumo do mês*\n"
        f"Total: *{moeda_fmt(total)}*\n\n"
        f"Por categoria:\n{fmt(by_cat)}\n\n"
        f"Por centro de custo:\n{fmt(by_cc)}"
    )
    await update.message.reply_markdown(msg)

async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, res = save_entry(update.effective_user.id, update.message.text or "")
    if ok:
        r = res
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("payment_method"): extras.append(f"💳 {r['payment_method']}")
        tail = ("\n" + " • ".join(extras)) if extras else ""
        await update.message.reply_text(
            f"✅ Lançado: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{tail}"
        )
    else:
        await update.message.reply_text(
            "Me manda algo tipo: 'paguei 200 no eletricista da *obra do Rodrigo*'\n"
            "ou usa /despesa 1200 tinta *reforma Joana*"
        )


# ============== ÁUDIO (WHISPER) ==============
async def voice_or_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not openai_client:
        await update.message.reply_text(
            "Transcrição de áudio não configurada. Defina OPENAI_API_KEY no Render. 😉"
        )
        return

    tgfile = None
    ext = ".audio"
    if update.message.voice:
        f = await update.message.voice.get_file()
        tgfile = f
        ext = ".oga"
    elif update.message.audio:
        f = await update.message.audio.get_file()
        tgfile = f
        mime = update.message.audio.mime_type or ""
        if "mpeg" in mime or "mp3" in mime:
            ext = ".mp3"
        elif "ogg" in mime:
            ext = ".ogg"
        elif "wav" in mime:
            ext = ".wav"
    else:
        await update.message.reply_text("Não recebi um áudio válido.")
        return

    local_path = f"/tmp/{uuid.uuid4().hex}{ext}"
    await tgfile.download_to_drive(local_path)

    try:
        def _transcribe(path: str) -> str:
            with open(path, "rb") as fh:
                resp = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=fh,
                    language="pt"
                )
            return getattr(resp, "text", "").strip()

        text_out = await asyncio.to_thread(_transcribe, local_path)

    except Exception as e:
        await update.message.reply_text(f"Não consegui transcrever o áudio. {e}")
        return
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass

    if not text_out:
        await update.message.reply_text("Não consegui entender o áudio.")
        return

    ok, res = save_entry(update.effective_user.id, text_out)
    if ok:
        r = res
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("payment_method"): extras.append(f"💳 {r['payment_method']}")
        tail = ("\n" + " • ".join(extras)) if extras else ""
        await update.message.reply_text(
            f"🗣️ Transcrito: “{text_out}”\n"
            f"✅ Lançado: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{tail}"
        )
    else:
        await update.message.reply_text(
            f"🗣️ Transcrito: “{text_out}”\n"
            f"⚠️ {res}"
        )


# ============== TELEGRAM APP ==============
tg_app: Application = ApplicationBuilder().token(TOKEN).build()

# Comandos
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("autorizar", cmd_autorizar))
tg_app.add_handler(CommandHandler("despesa", cmd_despesa))
tg_app.add_handler(CommandHandler("receita", cmd_receita))
tg_app.add_handler(CommandHandler("relatorio", cmd_relatorio))

# Texto comum
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))

# Áudio/Voice -> Whisper
tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_or_audio))


# ============== FASTAPI / WEBHOOK ==============
class TgUpdate(BaseModel):
    update_id: int | None = None

@app.on_event("startup")
async def on_startup():
    await tg_app.initialize()
    await tg_app.start()

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

@app.get("/")
def alive():
    return {"boris": "ok"}
