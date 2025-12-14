import os, re, datetime, asyncio
from collections import defaultdict
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dateutil.parser import parse as dateparse
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from supabase import create_client, Client

TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not (TOKEN and SUPABASE_URL and SUPABASE_KEY):
    raise RuntimeError("Faltam variáveis de ambiente: TELEGRAM_TOKEN, SUPABASE_URL, SUPABASE_KEY")

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

def money_from_text(txt:str):
    # Extrai primeiro valor monetário do texto (R$ 1.200,00 / 1200 / 1.200,00).
    s = txt.replace("R$", "").replace(" ", "")
    m = re.search(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,\d{2})?", s)
    if not m:
        return None
    raw = m.group(0).replace(".", "").replace(",", ".")
    try:
        return round(float(raw), 2)
    except:
        return None

def guess_type(txt:str):
    return "income" if re.search(r"\b(recebi|receita|entrada|vendi)\b", txt, re.I) else "expense"

CATEGORY_RULES = [
    (r"eletricist|el[eé]tric", "Elétrico"),
    (r"hidraul|hidráuli", "Hidráulico"),
    (r"drywall|gesso", "Drywall"),
    (r"pintur", "Pintura"),
    (r"ferra|parafus|broca", "Ferragens"),
    (r"tr[aá]fego|ads|google|meta|facebook|insta", "Marketing"),
    (r"aluguel|loca", "Custos Fixos"),
    (r"m[aã]o de obra|di[áa]ria|pedreir|ajudant", "Mão de Obra"),
    (r"bobcat|compactador|gerador|equip", "Equipamentos"),
    (r"uber|frete|log[íi]stic|combust", "Logística"),
]

def guess_category(txt:str):
    low = txt.lower()
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, low):
            return cat
    return "Outros"

def guess_cc(txt:str):
    m = re.search(r"\bbloco\s*([a-f])\b", txt, re.I)
    if m:
        return f"BLOCO_{m.group(1).upper()}"
    if re.search(r"\bsede|admin", txt, re.I):
        return "SEDE"
    return None

def get_or_none(res):
    # Helper para maybe_single do supabase-py: retornos com .data
    return res.data if hasattr(res, "data") else res

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    # cria ou atualiza registro do usuário como inativo por padrão
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
    # verifica se quem chamou é owner
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

def save_entry(tg_user_id:int, txt:str):
    amount = money_from_text(txt)
    if amount is None:
        return False, "Não achei o valor. Ex.: 'paguei 200 no eletricista'."
    etype = guess_type(txt)
    cat_name = guess_category(txt)
    cc_code = guess_cc(txt)

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

    # cost center id
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
        "created_by": user_id,
        "status": status
    }).execute()

    return True, {"amount": amount, "type": etype, "category": cat_name, "cc": cc_code, "status": status}

async def cmd_despesa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args) if context.args else (update.message.text or "")
    ok, res = save_entry(update.effective_user.id, txt)
    if ok:
        r = res
        await update.message.reply_text(
            f"✅ Lançado: R$ {r['amount']:.2f} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}"
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
    today = datetime.date.today()
    month_start = today.replace(day=1).isoformat()
    month_end = (today.replace(day=28) + datetime.timedelta(days=4)).replace(day=1).isoformat()

    # busca despesas do mês
    resp = sb.table("entries").select("amount,category_id,cost_center_id,type,entry_date,status").gte("entry_date", month_start).lt("entry_date", month_end).eq("type","expense").execute()
    rows = get_or_none(resp) or []

    # mapas auxiliares
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

async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, res = save_entry(update.effective_user.id, update.message.text or "")
    if ok:
        r = res
        await update.message.reply_text(
            f"✅ Lançado: R$ {r['amount']:.2f} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}"
        )
    else:
        await update.message.reply_text(
            "Me manda algo tipo: 'paguei 200 no eletricista do Bloco E'\n"
            "ou usa /despesa 1200 tráfego pago SEDE"
        )

# Telegram app
tg_app: Application = ApplicationBuilder().token(TOKEN).build()
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("autorizar", cmd_autorizar))
tg_app.add_handler(CommandHandler("despesa", cmd_despesa))
tg_app.add_handler(CommandHandler("receita", cmd_receita))
tg_app.add_handler(CommandHandler("relatorio", cmd_relatorio))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))

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
