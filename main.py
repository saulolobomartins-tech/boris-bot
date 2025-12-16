import os
import re
import uuid
import unicodedata
import datetime
from datetime import date, timedelta
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
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def moeda_fmt(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_or_none(res):
    return res.data if hasattr(res, "data") else res

# =====================================================================================
#                              MULTI-TENANT (account_id)
# =====================================================================================

_DEFAULT_ACCOUNT_ID = None

def get_default_account_id() -> str:
    """
    Busca (ou cria) uma conta padrão.
    Depende de tabela accounts(name, plan, id).
    """
    global _DEFAULT_ACCOUNT_ID
    if _DEFAULT_ACCOUNT_ID:
        return _DEFAULT_ACCOUNT_ID

    res = sb.table("accounts").select("id").eq("name", "DEFAULT ACCOUNT").limit(1).execute()
    rows = get_or_none(res) or []
    if rows:
        _DEFAULT_ACCOUNT_ID = rows[0]["id"]
        return _DEFAULT_ACCOUNT_ID

    ins = sb.table("accounts").insert({"name": "DEFAULT ACCOUNT", "plan": "free"}).execute()
    created = get_or_none(ins) or []
    if created:
        _DEFAULT_ACCOUNT_ID = created[0]["id"]
        return _DEFAULT_ACCOUNT_ID

    res2 = sb.table("accounts").select("id").eq("name", "DEFAULT ACCOUNT").limit(1).execute()
    rows2 = get_or_none(res2) or []
    if not rows2:
        raise RuntimeError("Não achei/criei a DEFAULT ACCOUNT em accounts.")
    _DEFAULT_ACCOUNT_ID = rows2[0]["id"]
    return _DEFAULT_ACCOUNT_ID


def ensure_category_id(account_id: str, name: str) -> str | None:
    if not name:
        return None
    try:
        r = sb.table("categories").select("id").eq("account_id", account_id).eq("name", name).limit(1).execute()
        rows = get_or_none(r) or []
        if rows:
            return rows[0]["id"]

        ins = sb.table("categories").insert({"account_id": account_id, "name": name}).execute()
        created = get_or_none(ins) or []
        if created:
            return created[0]["id"]

        r2 = sb.table("categories").select("id").eq("account_id", account_id).eq("name", name).limit(1).execute()
        rows2 = get_or_none(r2) or []
        return rows2[0]["id"] if rows2 else None
    except Exception:
        return None


def ensure_cost_center_id(account_id: str, code: str) -> str | None:
    if not code:
        return None
    try:
        r = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", code).limit(1).execute()
        rows = get_or_none(r) or []
        if rows:
            return rows[0]["id"]

        ins = sb.table("cost_centers").insert({"account_id": account_id, "code": code, "name": code}).execute()
        created = get_or_none(ins) or []
        if created:
            return created[0]["id"]

        r2 = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", code).limit(1).execute()
        rows2 = get_or_none(r2) or []
        return rows2[0]["id"] if rows2 else None
    except Exception:
        return None

# =====================================================================================
#                              REGRAS & DICIONÁRIOS
# =====================================================================================

CATEGORY_RULES = [
    # Mão de obra / serviços
    (r"\b(mao\s*de\s*obra|m[aã]o\s*de\s*obra|diari(a|as)|diaria(s)?|pedreir|ajudant|servente|marceneir|soldador|aplicador)\b", "Mão de Obra"),

    # Elétrica
    (r"\b(eletricist|eletric(a|o)?|fiao|fiacao|fiação|fio|disjuntor|quadro|tomad(a|as)|interruptor(es)?|spot|led|cabeamento)\b", "Elétrico"),

    # Hidráulica / hidrossanitário
    (r"\b(hidraul(ic|i|ica|ico)|hidrossanit(a|á)ri(o|a)|encanador|encanamento|encanar|cano(s)?|tubo(s)?|tubo\s*pex|registro|torneira|ralo|caixa\s*d'?agua|caixa\s*d'?água|esgoto|bomba)\b", "Hidráulico"),

    # Drywall / gesso
    (r"\b(drywall|forro|gesso|placa\s*acartonad)\b", "Drywall/Gesso"),

    # Pintura
    (r"\b(pintur(a|ar)|tinta(s)?|massa\s*corrida|lixa|rolo|fita\s*crepe|spray)\b", "Pintura"),

    # Estrutura / fundação / alvenaria
    (r"\b(fundacao|funda[cç][aã]o|sapata|broca|estaca|viga|pilar|laje|baldrame|cimento|areia|brita|argamassa|reboco|concreto|graute|bloco\s*ceram|vergalh|arma[cç][aã]o|forma)\b", "Estrutura/Alvenaria"),

    # Cobertura
    (r"\b(telha|calha|ruf(o|o)?|cumeeira|aluminio|zinco|manta\s*t[eé]rmica|termoac(o|ô)stic)\b", "Cobertura"),

    # Acabamento
    (r"\b(granito|porcelanato|piso|rodape|rodap[eé]|revestimento|rejunte|argamassa\s*col)\b", "Acabamento"),

    # Esquadrias / vidro
    (r"\b(porta(s)?|janela(s)?|vidro|esquadria|fechadur(a|as)|dobradi[cç]a|temperado|kit\s*porta)\b", "Esquadrias/Vidro"),

    # Impermeabilização
    (r"\b(impermeabiliza|manta\s*asf[aá]ltica|vedacit|sika)\b", "Impermeabilização"),

    # Ferragens / consumíveis
    (r"\b(ferra(gem|gens)?|parafus(o|os)|broca(s)?|eletrodo(s)?|disco\s*corte|abracadeira|abra[cç]adeira|chumbador|rebite)\b", "Ferragens/Consumíveis"),

    # Ferramentas
    (r"\b(ferramenta(s)?|esmerilhadeira|serra\s*circular|lixadeira|parafusadeira|multimetro|multímetro|trena)\b", "Ferramentas"),

    # Logística
    (r"\b(uber|frete|entrega|logistic(a|o)?|carretinha|transport(e|adora)?)\b", "Logística"),

    # Combustível
    (r"\b(combust(iv|í)vel|diesel|gasolina|etanol|oleo|óleo|lubrificante|posto)\b", "Combustível"),

    # Equipamentos
    (r"\b(bobcat|compactador|gerador|betoneira|aluguel\s*equip|loca[cç][aã]o\s*equip|munck|plataforma|guindaste)\b", "Equipamentos"),

    # Marketing
    (r"\b(trafego|tr[aá]fego|ads|google|meta|facebook|instagram|impulsionamento|an[uú]ncio)\b", "Marketing"),

    # Custos fixos
    (r"\b(aluguel|loca[cç][aã]o\s*de\s*sala|internet|energia|conta\s*de\s*luz|conta\s*de\s*agua|conta\s*de\s*água|agua|água|telefone|contabilidade|escritorio|escrit[oó]rio)\b", "Custos Fixos"),

    # Taxas / financeiro
    (r"\b(taxa|emolumento|cartorio|cartório|crea|art|multa|juros|tarifa|banco|ted\b|boleto|iof)\b", "Taxas/Financeiro"),

    # Alimentação (singular e plural + gírias)
    (r"\b(comida(s)?|refeic(a|ã)o|refei[cç][oõ]es|lanche(s)?|marmit(a|as)|quentinha(s)?|almo[cç]o(s)?|jantar(es)?|restaurante(s)?)\b", "Alimentação"),
]

DEFAULT_CATEGORY = "Outros"

PAYMENT_SYNONYMS = {
    "PIX":      r"\bpix\b|\bfiz\s+um\s+pix\b|\bmandei\s+um\s+pix\b|\bchave\s+pix\b",
    "CREDITO":  r"\bcr[eé]dito\b|\bno\s+cart[aã]o\s+de\s+cr[eé]dito\b|\bpassei\s+no\s+cr[eé]dito\b|\bpassei\s+no\s+cart[aã]o\b",
    "DEBITO":   r"\bd[eé]bito\b|\bno\s+cart[aã]o\s+de\s+d[eé]bito\b|\bpassei\s+no\s+d[eé]bito\b",
    "DINHEIRO": r"\bdinheiro\b|\bem\s+esp[eé]cie\b|\bcash\b",
    "VALE":     r"\bvale\b|\badiantamento\b",
}

MONTHS_PT = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12
}

QUERY_INTENT_RE = re.compile(
    r"\b(quanto\s+(eu\s+)?gastei|gastos|relat[oó]rio|me\s+mostra|mostra\s+pra\s+mim|me\s+manda)\b",
    re.I
)

# =====================================================================================
#                               PARSE DE TEXTO
# =====================================================================================

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
    Retorna: OBRA_RODRIGO / REFORMA_JOANA / CONTAINER_CASTANHAL
    """
    t = _norm(txt)
    m = re.search(r"\b(obra|reforma|container)\s+(?:do|da|de)?\s+([a-z0-9][a-z0-9\s\-_.]+)\b", t)
    if m:
        tipo = m.group(1)
        nome = m.group(2).strip()
        nome = re.split(
            r"\b(por|pra|pro|no|na|em|para|paguei|gastei|comprei|recebi|pix|credito|crédito|debito|débito|cartao|cartão)\b",
            nome
        )[0].strip()
        if nome:
            return f"{tipo.upper()}_{_slugify_name(nome)}"
    return None

def parse_date_pt(txt: str) -> str | None:
    t = _norm(txt)
    today = date.today()

    if "hoje" in t: return today.isoformat()
    if "ontem" in t: return (today - timedelta(days=1)).isoformat()
    if "anteontem" in t: return (today - timedelta(days=2)).isoformat()
    if "amanha" in t: return (today + timedelta(days=1)).isoformat()

    if "semana passada" in t:
        wd = today.weekday()
        last_monday = today - timedelta(days=wd + 7)
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
        mes = m.group(1).replace("ç", "c")
        ano = int(m.group(2)) if m.group(2) else today.year
        month_num = MONTHS_PT.get(mes, None)
        if month_num:
            s = date(ano, month_num, 1)
            e = (s.replace(day=28) + timedelta(days=4)).replace(day=1)
            return s.isoformat(), e.isoformat(), f"{m.group(1).title()} {ano}"

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
    return bool(re.search(r"\b(entrou|recebi|receitas?|quanto\s+entrou)\b", text or "", re.I))

def is_report_intent(text: str):
    return bool(QUERY_INTENT_RE.search(text or ""))

# =====================================================================================
#                               CC STATE (last_cc) + pending
# =====================================================================================

PENDING_BY_USER: dict[int, dict] = {}

def _get_user_row(tg_user_id: int) -> dict | None:
    r = sb.table("users").select("id,role,is_active,account_id").eq("tg_user_id", tg_user_id).limit(1).execute()
    rows = get_or_none(r) or []
    return rows[0] if rows else None

def _get_last_cc(tg_user_id: int) -> str | None:
    try:
        r = sb.table("users").select("last_cc").eq("tg_user_id", tg_user_id).limit(1).execute()
        rows = get_or_none(r) or []
        if rows and rows[0].get("last_cc"):
            return rows[0]["last_cc"]
    except Exception:
        pass
    return None

def _set_last_cc(tg_user_id: int, cc_code: str):
    try:
        sb.table("users").update({"last_cc": cc_code}).eq("tg_user_id", tg_user_id).execute()
    except Exception:
        pass

# =====================================================================================
#                               PERSISTÊNCIA
# =====================================================================================

def save_entry(tg_user_id: int, txt: str, force_cc: str | None = None):
    """
    Blindado: nunca deixa exception subir e matar o reply.
    """
    try:
        low = _norm(txt)

        amount = money_from_text(txt)
        if amount is None:
            return False, "Não achei o valor. Ex.: 'paguei 200 no eletricista'."

        # força INCOME com termos de entrada
        if re.search(r"\b(recebi|receita|entrada|entrou|vendi|aluguel\s+recebid|pagaram|pagou\s*pra\s*mim)\b", low):
            etype = "income"
        else:
            etype = "expense"

        user_row = _get_user_row(tg_user_id)
        if not user_row or not user_row.get("is_active"):
            return False, "Usuário não autorizado. Use /start e peça autorização."

        user_id = user_row["id"]
        role = user_row["role"]
        account_id = user_row.get("account_id") or get_default_account_id()

        cat_name = guess_category(txt)
        paid_via = guess_payment(txt)
        dtx = parse_date_pt(txt)
        entry_date = dtx or datetime.date.today().isoformat()

        cc_code = force_cc or guess_cc(txt)

        used_last_cc = False
        if not cc_code:
            last_cc = _get_last_cc(tg_user_id)
            if last_cc:
                cc_code = last_cc
                used_last_cc = True

        cat_id = ensure_category_id(account_id, cat_name)
        cc_id = ensure_cost_center_id(account_id, cc_code) if cc_code else None

        status = "approved" if role in ("owner", "partner") else "pending"

        payload = {
            "account_id": account_id,
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

        if cc_code:
            _set_last_cc(tg_user_id, cc_code)

        return True, {
            "amount": amount,
            "type": etype,
            "category": cat_name,
            "cc": cc_code,
            "status": status,
            "paid_via": paid_via,
            "entry_date": entry_date,
            "used_last_cc": used_last_cc
        }

    except Exception as e:
        return False, f"Erro interno no save_entry: {type(e).__name__}: {e}"

# =====================================================================================
#                               CONSULTAS / RELATÓRIOS
# =====================================================================================

async def run_query_and_reply(update: Update, text: str):
    # pega account_id do usuário que pediu relatório
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return
    account_id = user_row.get("account_id") or get_default_account_id()

    start, end, label = parse_period_pt(text)
    cat = guess_category_filter(text)
    paid = guess_paid_filter(text)
    cc_code = guess_cc_filter(text)
    is_income = is_income_query(text)

    q = sb.table("entries").select("amount,category_id,cost_center_id,paid_via,type,entry_date") \
        .eq("account_id", account_id) \
        .gte("entry_date", start).lt("entry_date", end)

    q = q.eq("type", "income" if is_income else "expense")

    if paid:
        q = q.eq("paid_via", paid)

    if cat:
        c = sb.table("categories").select("id").eq("account_id", account_id).eq("name", cat).limit(1).execute()
        cd = get_or_none(c) or []
        if cd:
            q = q.eq("category_id", cd[0]["id"])

    if cc_code:
        cc = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", cc_code).limit(1).execute()
        ccd = get_or_none(cc) or []
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
#                               PROCESSAMENTO ÚNICO (texto e áudio)
# =====================================================================================

async def process_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    try:
        uid = update.effective_user.id
        user_text = (user_text or "").strip()
        if not user_text:
            return

        # pendência esperando CC
        if uid in PENDING_BY_USER:
            cc = guess_cc(user_text)
            if cc:
                pending = PENDING_BY_USER.pop(uid)
                ok, res = save_entry(uid, pending["txt"], force_cc=cc)
                if ok:
                    r = res
                    await update.message.reply_text(
                        f"✅ Lançado (com CC informado): {moeda_fmt(r['amount'])} • {r['category']} • {r['cc']} • {r['status']}"
                    )
                else:
                    await update.message.reply_text(f"⚠️ {res}")
                return

        # seta CC sem valor
        cc_only = guess_cc(user_text)
        if cc_only and money_from_text(user_text) is None and not is_report_intent(user_text):
            _set_last_cc(uid, cc_only)
            await update.message.reply_text(f"✅ Obra/CC atual definido: {cc_only}")
            return

        # relatório
        if is_report_intent(user_text):
            await run_query_and_reply(update, user_text)
            return

        # se não tem CC nem last_cc, pede confirmação
        cc_in_text = guess_cc(user_text)
        last_cc = _get_last_cc(uid)

        if (not cc_in_text) and (not last_cc) and money_from_text(user_text) is not None:
            PENDING_BY_USER[uid] = {"txt": user_text}
            await update.message.reply_text(
                "Beleza. Só me diz **qual obra/centro de custo** pra eu lançar certinho.\n"
                "Ex: `obra do Rodrigo` ou `reforma da Ellen`.\n\n"
                "Dica: define a obra do dia com `/obra Rodrigo`."
            )
            return

        ok, res = save_entry(uid, user_text)
        if ok:
            r = res
            label = "Receita" if r.get("type") == "income" else "Lançado"
            extras = []
            if r.get("entry_date"): extras.append(f"🗓️ {r['entry_date']}")
            if r.get("paid_via"): extras.append(f"💳 {r['paid_via']}")
            if r.get("used_last_cc") and r.get("cc"): extras.append(f"📌 CC assumido: {r['cc']}")
            tail = ("\n" + " • ".join(extras)) if extras else ""

            # dica pro usuário quando assumiu CC
            hint = ""
            if r.get("used_last_cc") and r.get("cc"):
                hint = "\nSe essa obra não for a certa, manda: `obra do <nome>` (que eu ajusto pro próximo)."

            await update.message.reply_text(
                f"✅ {label}: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'} • {r['status']}{tail}{hint}"
            )
        else:
            await update.message.reply_text(f"⚠️ {res}")

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no processamento: {type(e).__name__}: {e}")

# =====================================================================================
#                               TELEGRAM HANDLERS
# =====================================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    default_account_id = get_default_account_id()

    exist = sb.table("users").select("*").eq("tg_user_id", u.id).execute()
    data = get_or_none(exist) or []

    if not data:
        sb.table("users").insert({
            "tg_user_id": u.id,
            "name": u.full_name,
            "role": "viewer",
            "is_active": False,
            "account_id": default_account_id,
            "last_cc": None
        }).execute()
    else:
        upd = {}
        if not data[0].get("account_id"):
            upd["account_id"] = default_account_id
        if "last_cc" not in data[0]:
            upd["last_cc"] = None
        if upd:
            sb.table("users").update(upd).eq("tg_user_id", u.id).execute()

    await update.message.reply_text(
        f"Fala, {u.first_name}! Eu sou o Boris.\n"
        f"Teu Telegram user id é: {u.id}\n"
        f"Pede pro owner te autorizar com /autorizar {u.id} role=buyer"
    )

async def cmd_autorizar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    you = _get_user_row(u.id)

    if not you or you.get("role") != "owner" or not you.get("is_active"):
        await update.message.reply_text("Somente o owner pode autorizar usuários.")
        return

    if len(context.args) == 0:
        await update.message.reply_text("Uso: /autorizar <tg_user_id> role=owner|partner|buyer|viewer")
        return

    target = int(context.args[0])
    role = "buyer"
    for a in context.args[1:]:
        if a.startswith("role="):
            role = a.split("=", 1)[1].strip()

    owner_account_id = you.get("account_id") or get_default_account_id()

    sb.table("users").upsert({
        "tg_user_id": target,
        "role": role,
        "is_active": True,
        "name": "",
        "account_id": owner_account_id
    }).execute()

    await update.message.reply_text(f"Usuário {target} autorizado como {role} ✅")

async def cmd_obra(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /obra Rodrigo  -> define CC atual (OBRA_RODRIGO)
    /obra Ellen    -> define CC atual (OBRA_ELLEN)
    """
    name = " ".join(context.args).strip()
    if not name:
        await update.message.reply_text("Uso: /obra <nome>. Ex: /obra Rodrigo")
        return
    cc = f"OBRA_{_slugify_name(name)}"
    _set_last_cc(update.effective_user.id, cc)
    await update.message.reply_text(f"✅ Obra/CC atual definido:\n{cc}")

async def cmd_despesa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args).strip()
    if not txt:
        await update.message.reply_text("Uso: /despesa <texto>. Ex: /despesa 200 eletricista pix obra do Rodrigo")
        return
    await process_user_text(update, context, txt)

async def cmd_receita(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args).strip()
    if not txt:
        await update.message.reply_text("Uso: /receita <texto>. Ex: /receita 1200 pagamento Joana pix")
        return
    await process_user_text(update, context, "receita " + txt)

async def cmd_relatorio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Resumo do mês por categoria e CC (despesas) filtrando por conta.
    """
    try:
        user_row = _get_user_row(update.effective_user.id)
        if not user_row or not user_row.get("is_active"):
            await update.message.reply_text("Usuário não autorizado.")
            return
        account_id = user_row.get("account_id") or get_default_account_id()

        today = datetime.date.today()
        month_start = today.replace(day=1).isoformat()
        month_end = (today.replace(day=28) + timedelta(days=4)).replace(day=1).isoformat()

        resp = sb.table("entries").select(
            "amount,category_id,cost_center_id,type,entry_date,status"
        ).eq("account_id", account_id) \
         .gte("entry_date", month_start).lt("entry_date", month_end) \
         .eq("type", "expense").execute()

        rows = get_or_none(resp) or []

        cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
        ccs_rows  = get_or_none(sb.table("cost_centers").select("id,code").eq("account_id", account_id).execute()) or []
        cats = {r["id"]: r["name"] for r in cats_rows}
        ccs  = {r["id"]: r["code"] for r in ccs_rows}

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

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no /relatorio: {type(e).__name__}: {e}")

# -------------------- TEXTO --------------------
async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_user_text(update, context, update.message.text or "")

# -------------------- ÁUDIO (voice/audio) — robusto + debug --------------------
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not oa_client:
        await update.message.reply_text("Whisper não está configurado (OPENAI_API_KEY ausente).")
        return

    try:
        tgfile = None
        ext = ".audio"

        if update.message.voice:
            tgfile = await update.message.voice.get_file()
            ext = ".oga"
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
            try:
                with open(local_path, "rb") as fh:
                    resp = oa_client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=fh,
                        language="pt"
                    )
                text_out = (getattr(resp, "text", "") or "").strip()
            except Exception:
                with open(local_path, "rb") as fh:
                    resp = oa_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=fh,
                        language="pt"
                    )
                text_out = (getattr(resp, "text", "") or "").strip()
        finally:
            try:
                os.remove(local_path)
            except Exception:
                pass

        if not text_out:
            await update.message.reply_text("Não consegui entender o áudio.")
            return

        await update.message.reply_text(f"🗣️ Transcrito: “{text_out}”")

        # Processa como texto (e se der erro, aparece no chat)
        await process_user_text(update, context, text_out)

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no handle_audio: {type(e).__name__}: {e}")

# =====================================================================================
#                               TELEGRAM APP
# =====================================================================================

tg_app: Application = ApplicationBuilder().token(TOKEN).build()

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("autorizar", cmd_autorizar))
tg_app.add_handler(CommandHandler("obra", cmd_obra))
tg_app.add_handler(CommandHandler("despesa", cmd_despesa))
tg_app.add_handler(CommandHandler("receita", cmd_receita))
tg_app.add_handler(CommandHandler("relatorio", cmd_relatorio))

tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))
tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

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
