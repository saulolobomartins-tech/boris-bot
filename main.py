import os
import re
import io
import uuid
import unicodedata
import asyncio
import datetime
from datetime import date, timedelta, datetime as dt
from collections import defaultdict

from fastapi import FastAPI, Request
from pydantic import BaseModel

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
oa_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
#                              NORMALIZAÇÃO / HELPERS
# =====================================================================================

def _norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def moeda_fmt(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_or_none(res):
    return res.data if hasattr(res, "data") else res

# =====================================================================================
#                              REGRAS & DICIONÁRIOS
# =====================================================================================

# Categorias ampliadas (regex -> nome)
CATEGORY_RULES = [
    # Mão de obra e serviços
    (r"\bma(o|ao)\s*de\s*obra|diaria|diaria(s)?|pedreir|ajudant|servente|marceneir|soldador|aplicador", "Mão de Obra"),
    # Elétrica / Hidráulica / Drywall / Pintura
    (r"eletricist|eletric|fio|disjuntor|quadro|tomada|interruptor|spot|led|cabeamento|fiação", "Elétrico"),
    (r"hidraul|hidrauli|cano|tubo pex|regist|torneira|ralo|caixa d'?agua|esgoto|bomba|hidr[aá]ul", "Hidráulico"),
    (r"drywall|forro|gesso|placa acartonad", "Drywall/Gesso"),
    (r"pintur|tinta|massa corrida|lixa|rolo|fita crepe|spray", "Pintura"),
    # Estrutura / Cobertura
    (r"cimento|areia|brita|argamassa|reboco|concreto|graute|bloco ceram|vergalh|arma[cç][aã]o|forma", "Estrutura/Alvenaria"),
    (r"telha|calha|ruf|cumeeira|aluminio|zinco|manta t[eé]rmica|termoac[oô]stic", "Cobertura"),
    # Acabamento / Esquadrias
    (r"granito|porcelanato|piso|rodape|rodap[eé]|revestimento|rejunte|argamassacol", "Acabamento"),
    (r"porta|janela|vidro|esquadria|fechadur|dobradic|dobradi[cç]a|temperado|kit porta", "Esquadrias/Vidro"),
    # Impermeabilização
    (r"impermeabiliza|manta asf[aá]ltica|vedacit|sika", "Impermeabilização"),
    # Ferragens / Ferramentas
    (r"ferra|parafus|broca|eletrodo|disco corte|abracadeira|abra[cç]adeira|chumbador|rebite", "Ferragens/Consumíveis"),
    (r"ferramenta|esmerilhadeira|serra circular|lixadeira|parafusadeira|multimetro|trena", "Ferramentas"),
    # Logística e equipamentos
    (r"uber|frete|entrega|logistic|carretinha|transport", "Logística"),
    (r"combust|diesel|gasolina|etanol|oleo|óleo|lubrificante|posto", "Combustível"),
    (r"bobcat|compactador|gerador|betoneira|aluguel equip|loca[cç][aã]o equip|munck|plataforma|guindaste", "Equipamentos"),
    # Adm/marketing/financeiro
    (r"trafego|tr[aá]fego|ads|google|meta|facebook|instagram|impulsionamento|an[uú]ncio", "Marketing"),
    (r"aluguel|loca[cç][aã]o de sala|internet|energia|conta de luz|conta de agua|água|telefone|contabilidade|escritorio|escrit[oó]rio", "Custos Fixos"),
    (r"taxa|emolumento|cartorio|cartório|crea|art|multa|juros|tarifa|banco|ted\b|boleto|iof", "Taxas/Financeiro"),
    # Alimentação
    (r"comida|refei[cç][aã]o|lanche|marmit|almo[cç]o|jantar|restaurante", "Alimentação"),
]
DEFAULT_CATEGORY = "Outros"

# Formas de pagamento (sinônimos)
PAYMENT_SYNONYMS = {
    "PIX":      r"\bpix\b|\bfiz um pix\b|\bmandei um pix\b|\bchave pix\b",
    "CREDITO":  r"\bcr[eé]dito\b|\bno cart[aã]o de cr[eé]dito\b|\bpassei no cr[eé]dito\b|\bpassei no cart[aã]o\b",
    "DEBITO":   r"\bd[eé]bito\b|\bno cart[aã]o de d[eé]bito\b|\bpassei no d[eé]bito\b",
    "DINHEIRO": r"\bdinheiro\b|\bem esp[eé]cie\b|\bcash\b",
    "VALE":     r"\bvale\b|\bvale(i|u)\b|\badiantamento\b",
}

# Meses PT
MONTHS_PT = {
    "janeiro":1, "fevereiro":2, "marco":3, "março":3, "abril":4, "maio":5, "junho":6,
    "julho":7, "agosto":8, "setembro":9, "outubro":10, "novembro":11, "dezembro":12
}

# Intent para consultas/relatórios
QUERY_INTENT_RE = re.compile(
    r"\b(quanto\s+(eu\s+)?gastei|gastos|relat[oó]rio|me\s+mostra|mostra\s+pra\s+mim|me\s+manda)\b",
    re.I
)

# =====================================================================================
#                               PARSE DE TEXTO
# =====================================================================================

def money_from_text(txt:str):
    s = _norm(txt).replace("r$", "").replace(" ", "")
    m = re.search(r"(-?\d{1,3}(?:\.\d{3})+|-?\d+)(?:,\d{2})?", s)
    if not m:
        return None
    raw = m.group(0).replace(".", "").replace(",", ".")
    try:
        return round(float(raw), 2)
    except:
        return None

def guess_type(txt: str):
    t = _norm(txt)
    if re.search(r"\b(recebi|receita|entrada|vendi|entrou|aluguel recebido)\b", t):
        return "income"
    return "expense"

def guess_payment(txt: str):
    low = _norm(txt)
    for label, pat in PAYMENT_SYNONYMS.items():
        if re.search(pat, low):
            return label
    return None

def guess_category(txt: str):
    low = _norm(txt)
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, low):
            return cat
    return DEFAULT_CATEGORY

def _slugify_name(name: str) -> str:
    s = _norm(name)
    s = re.sub(r"[^a-z0-9]+", "", s).strip("")
    return s.upper()

def guess_cc(txt: str) -> str | None:
    """
    Reconhece: 'obra do rodrigo', 'reforma da joana', 'container de castanhal', etc.
    Retorna code padronizado: OBRA_RODRIGO / REFORMA_JOANA / CONTAINER_CASTANHAL
    """
    t = _norm(txt)
    m = re.search(r"\b(obra|reforma|container)\s+(?:do|da|de)?\s+([a-z0-9][a-z0-9\s\-_.]+)\b", t)
    if m:
        tipo = m.group(1)
        nome = m.group(2).strip()
        nome = re.split(r"\b(por|pra|pro|no|na|em|para|paguei|gastei|comprei|recebi|pix|credito|debito|cartao|cartão)\b", nome)[0].strip()
        if nome:
            return f"{tipo.upper()}_{_slugify_name(nome)}"
    return None

def _ensure_cost_center(code: str) -> int | None:
    try:
        res = sb.table("cost_centers").select("id").eq("code", code).execute()
        rows = get_or_none(res) or []
        if rows:
            return rows[0]["id"]
        ins = sb.table("cost_centers").insert({"code": code, "name": code}).execute()
        created = get_or_none(ins) or []
        if created:
            return created[0]["id"]
        res2 = sb.table("cost_centers").select("id").eq("code", code).execute()
        rows2 = get_or_none(res2) or []
        return rows2[0]["id"] if rows2 else None
    except Exception:
        return None

def parse_date_pt(txt: str) -> str | None:
    """
    Datas naturais -> YYYY-MM-DD
    hoje, ontem, anteontem, amanhã
    'dia 12' (do mês atual)
    12/10(/2025) ou 12-10(-2025)
    dias da semana (pega a última ocorrência)
    mês passado
    """
    t = _norm(txt)
    today = date.today()

    if "hoje" in t: return today.isoformat()
    if "ontem" in t: return (today - timedelta(days=1)).isoformat()
    if "anteontem" in t: return (today - timedelta(days=2)).isoformat()
    if "amanha" in t: return (today + timedelta(days=1)).isoformat()
    if "semana passada" in t:
        wd = today.weekday()
        last_monday = today - timedelta(days=wd+7)
        return last_monday.isoformat()
    if "mes passado" in t or "mês passado" in txt:
        first = today.replace(day=1)
        last_month = (first - timedelta(days=1)).replace(day=1)
        return last_month.isoformat()

    week_map = {
        "segunda": 0, "terca": 1, "terça": 1, "quarta": 2,
        "quinta": 3, "sexta": 4, "sabado": 5, "sábado": 5, "domingo": 6
    }
    for k, wd_target in week_map.items():
        if k in t:
            wd_today = today.weekday()
            delta = (wd_today - wd_target) % 7
            return (today - timedelta(days=delta)).isoformat()

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
            return date(y, mo, d).isoformat()
        except ValueError:
            pass

    m2 = re.search(r"\bdia\s+(\d{1,2})\b", t)
    if m2:
        d = int(m2.group(1))
        try:
            return date(today.year, today.month, d).isoformat()
        except ValueError:
            pass

    return None

def _first_day_of_week(d: date):
    return d - timedelta(days=d.weekday())

def _last_day_of_week(d: date):
    return _first_day_of_week(d) + timedelta(days=7)

def parse_period_pt(text: str):
    """
    Retorna (start_date_iso, end_date_iso_exclusive, label)
    """
    low = _norm(text)
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
    low = _norm(text)
    for pat, name in CATEGORY_RULES:
        if re.search(pat, low):
            return name
    return None

def guess_paid_filter(text: str):
    low = _norm(text)
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
    # Normaliza
    low = _norm(txt)

    # Detecta valor
    amount = money_from_text(txt)
    if amount is None:
        return False, "Não achei o valor. Ex.: 'paguei 200 no eletricista'."

    # Tipo (força income se tiver palavras de entrada)
    if re.search(r"\b(recebi|receita|entrada|entrou|vendi|aluguel\s+recebid|pagaram|pagou\s*pra\s*mim)\b", low):
        etype = "income"
    else:
        etype = "expense"

    # Categorias / CC / Pagamento / Data
    cat_name = guess_category(txt)
    cc_code  = guess_cc(txt)
    paid_via = guess_payment(txt)
    dtx = parse_date_pt(txt)
    entry_date = dtx or datetime.date.today().isoformat()

    # Usuário
    u = sb.table("users").select("*").eq("tg_user_id", tg_user_id).execute()
    ud = get_or_none(u)
    if not ud or not ud[0]["is_active"]:
        return False, "Usuário não autorizado. Use /start e peça autorização."
    user_id = ud[0]["id"]
    role = ud[0]["role"]

    # Categoria
    c = sb.table("categories").select("id").eq("name", cat_name).execute()
    cd = get_or_none(c)
    cat_id = cd[0]["id"] if cd else None

    # Centro de custo
    cc_id = None
    if cc_code:
        cc_id = _ensure_cost_center(cc_code)

    status = "approved" if role in ("owner","partner") else "pending"

    payload = {
        "entry_date": entry_date,
        "type": etype,
        "amount": amount,
        "description": txt,
        "category_id": cat_id,
        "cost_center_id": cc_id,
        "paid_via": paid_via,
        "created_by": user_id,
        "status": status
    }

    try:
        sb.table("entries").insert(payload).execute()
    except Exception:
        payload.pop("paid_via", None)
        sb.table("entries").insert(payload).execute()

    return True, {
        "amount": amount,
        "type": etype,
        "category": cat_name,
        "cc": cc_code,
        "status": status,
        "paid_via": paid_via,
        "entry_date": entry_date
    }


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

    filtros = []
    if cat: filtros.append(cat)
    if paid: filtros.append(paid.title())
    if cc_code: filtros.append(cc_code)
    filtros_txt = f" | Filtros: {', '.join(filtros)}" if filtros else ""

    await update.message.reply_text(
        f"📊 Total de {'receitas' if is_income else 'gastos'} em {label}{filtros_txt}:\n*{moeda_fmt(total)}*",
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
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("paid_via"): extras.append(f"💳 {r['paid_via']}")
        tail = ("\n" + " • ".join(extras)) if extras else ""
        await update.message.reply_text(
            f"✅ Lançado: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{tail}"
        )
    else:
        await update.message.reply_text(f"⚠️ {res}")

async def cmd_receita(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args) if context.args else (update.message.text or "")
    ok, res = save_entry(update.effective_user.id, "receita "+txt)
    if ok:
        r = res
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("paid_via"): extras.append(f"💳 {r['paid_via']}")
        tail = ("\n" + " • ".join(extras)) if extras else ""
        await update.message.reply_text(
            f"✅ Receita: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'}{tail}"
        )
    else:
        await update.message.reply_text(f"⚠️ {res}")

async def cmd_relatorio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = datetime.date.today()
    month_start = today.replace(day=1).isoformat()
    month_end = (today.replace(day=28) + timedelta(days=4)).replace(day=1).isoformat()

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
        f"📊 Resumo do mês\n"
        f"Total: {moeda_fmt(total)}\n\n"
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
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("paid_via"): extras.append(f"💳 {r['paid_via']}")
        tail = ("\n" + " • ".join(extras)) if extras else ""
        await update.message.reply_text(
            f"✅ Lançado: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{tail}"
        )
    else:
        await update.message.reply_text(
            "Me manda algo tipo: 'paguei 200 no eletricista da obra do Rodrigo (pix)'\n"
            "ou usa /despesa 1200 tinta reforma Joana"
        )

# -------------------- ÁUDIO (voice/audio) — robusto com /tmp --------------------
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not oa_client:
        await update.message.reply_text("Whisper não está configurado (OPENAI_API_KEY ausente).")
        return

    tgfile = None
    ext = ".audio"

    if update.message.voice:
        tgfile = await update.message.voice.get_file()
        ext = ".oga"  # Telegram voice é OGG/Opus
    elif update.message.audio:
        tgfile = await update.message.audio.get_file()
        mime = (update.message.audio.mime_type or "").lower()
        if "mpeg" in mime or "mp3" in mime:
            ext = ".mp3"
        elif "ogg" in mime:
            ext = ".ogg"
        elif "wav" in mime:
            ext = ".wav"
        else:
            ext = ".audio"
    else:
        await update.message.reply_text("Não recebi um áudio válido.")
        return

    local_path = f"/tmp/{uuid.uuid4().hex}{ext}"
    await tgfile.download_to_drive(local_path)

    try:
        text_out = None
        # Tenta o modelo rápido primeiro
        try:
            with open(local_path, "rb") as fh:
                resp = oa_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=fh,
                    language="pt"
                )
            text_out = getattr(resp, "text", "") or ""
            text_out = text_out.strip()
        except Exception:
            # Fallback para whisper-1 (estável)
            with open(local_path, "rb") as fh:
                resp = oa_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=fh,
                    language="pt"
                )
            text_out = getattr(resp, "text", "") or ""
            text_out = text_out.strip()
    except Exception as e:
        await update.message.reply_text(f"Não consegui transcrever o áudio. Erro: {e}")
        try:
            os.remove(local_path)
        except Exception:
            pass
        return
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass

    if not text_out:
        await update.message.reply_text("Não consegui entender o áudio.")
        return

    # Consulta?
    if is_report_intent(text_out):
        await update.message.reply_text(f"🗣️ Transcrito: “{text_out}”")
        await run_query_and_reply(update, text_out)
        return

    # Lançamento padrão
    ok, res = save_entry(update.effective_user.id, text_out)
    if ok:
        r = res
        extras = []
        if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
        if r.get("paid_via"): extras.append(f"💳 {r['paid_via']}")
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
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))
tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

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
