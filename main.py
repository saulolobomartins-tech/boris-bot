import os
import re
import uuid
import unicodedata
import datetime
import asyncio
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

oa_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=35.0,
    max_retries=2
) if OPENAI_API_KEY else None

# -------------- ENV ----------------
TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or os.getenv("SUPABASE_KEY")

if not TOKEN:
    raise RuntimeError("Falta a variável de ambiente: TELEGRAM_TOKEN")

if not SUPABASE_URL:
    raise RuntimeError("Falta a variável de ambiente: SUPABASE_URL")

if not SUPABASE_KEY:
    raise RuntimeError(
        "Falta a variável de ambiente do Supabase. "
        "Defina SUPABASE_SERVICE_ROLE_KEY (recomendado) ou SUPABASE_KEY."
    )

if SUPABASE_KEY.startswith("sb_publishable_"):
    raise RuntimeError(
        "A chave informada é publishable/public. "
        "Para o Boris no backend, use a SECRET KEY do Supabase "
        "(sb_secret_...) em SUPABASE_SERVICE_ROLE_KEY."
    )

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------- FASTAPI --------------
app = FastAPI()

# =====================================================================================
#                          KEEPALIVE (Supabase Free) + Render
# =====================================================================================

_KEEPALIVE_TASK = None


async def _supabase_keepalive_once():
    try:
        sb.table("accounts").select("id").limit(1).execute()
        return True
    except Exception:
        return False


async def _daily_keepalive_loop():
    await asyncio.sleep(10)
    while True:
        try:
            await _supabase_keepalive_once()
        except Exception:
            pass
        await asyncio.sleep(24 * 60 * 60)

# =====================================================================================
#                              NORMALIZAÇÃO / HELPERS
# =====================================================================================


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def moeda_fmt(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def get_or_none(res):
    return res.data if hasattr(res, "data") else res


def data_fmt_out(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        d = datetime.date.fromisoformat(iso)
        return d.strftime("%d/%m/%Y")
    except Exception:
        return iso


def entry_emoji(etype: str | None) -> str:
    return "✅" if etype == "income" else "⛔"


def entry_label(etype: str | None) -> str:
    return "Receita" if etype == "income" else "Despesa"


def _clip(s: str, n: int = 40) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n - 1] + "…"


TG_SAFE_CHUNK = 3600


async def _send_long(update: Update, text: str):
    text = text or ""
    if len(text) <= TG_SAFE_CHUNK:
        await update.message.reply_text(text)
        return

    lines = text.split("\n")
    buff = ""
    for ln in lines:
        if len(buff) + len(ln) + 1 > TG_SAFE_CHUNK:
            await update.message.reply_text(buff)
            buff = ln
        else:
            buff = (buff + "\n" + ln) if buff else ln
    if buff:
        await update.message.reply_text(buff)

# =====================================================================================
#                              MULTI-TENANT (account_id)
# =====================================================================================

_DEFAULT_ACCOUNT_ID = None


def get_default_account_id() -> str:
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
    (r"\b("
     r"mao\s*de\s*obra|m[aã]o\s*de\s*obra|"
     r"diari(a|as)|di[aá]ria(s)?|"
     r"mestre\s+de\s+obras|encarregado|apontador|"
     r"pedreir(o|a)s?|"
     r"carpinteir(o|a)s?|"
     r"armador(es)?|"
     r"servente(s)?|"
     r"ajudant(e|es)|auxiliar(es)?|ajudante\s+geral|"
     r"pintor(es)?|"
     r"eletricist(a|o)?s?|"
     r"encanador(es)?|bombeiro\s*hidraulico|bombeiro\s*hidráulico|"
     r"gesseir(o|a)s?|"
     r"drywall|montador(es)?|"
     r"marceneir(o|a)s?|"
     r"serralheir(o|a)s?|serralheiro(s)?|"
     r"soldador(es)?|soldador(a)?s?|"
     r"vidraceir(o|a)s?|"
     r"azulejist(a|o)s?|"
     r"projetista(s)?|"
     r"topograf(o|a)|top[oó]graf(o|a)|"
     r"tecnico\s+de\s+seguran(c|ç)a|"
     r"tecnico\b|t[eé]cnic(o|a)s?|"
     r"engenheir(o|a)s?|"
     r"arquit(et|e)t(o|a)s?|"
     r"operador(es)?\s+de\s+maquina|operador(es)?\s+de\s+m[aá]quina|"
     r"operador(es)?\s+de\s+betoneira|"
     r"operador(es)?\s+de\s+compactador|"
     r"operador(es)?\s+de\s+munck|"
     r"montagem|instal(a|aç)[aã]o"
     r")\b", "Mão de Obra"),

    (r"\b("
     r"lazer|bar|cerveja|churrasco|restaurante|pizza|cinema|show|balada|happy\s*hour|"
     r"bebida(s)?|drink(s)?"
     r")\b", "Lazer"),

    (r"\b("
     r"uber|frete|entrega|logistic(a|o)?|transport(e|adora)?|"
     r"carretinha|carrocinha|reboque|"
     r"munk|munck|"
     r"pedagio|pedagios|ped[aá]gio|ped[aá]gios|"
     r"balsa|travessia\s+de\s+balsa|"
     r"pneu|pneus|"
     r"borracheiro|borracharia|calibragem|"
     r"manutencao|manuten[cç][aã]o|"
     r"peca|pecas|pe[cç]a|pe[cç]as|"
     r"oficina|"
     r"lavagem|lava\s*jato|lava\s+a\s*jato|lava-jato|"
     r"etios|tcross|t-cross|t\s*cross"
     r")\b", "Logística"),

    (r"\b(refriger(a|aç)[aã]o|ar\s*condicionado|split|vrf|central\s+de\s+ar|tubula(c|ç)[aã]o|linha\s+de\s+cobre|cobre\s+para\s+ar|gas\s+refrigerante|flange|vacuometro|manifold)\b", "Refrigeração"),
    (r"\b(abastec|combust(iv|í)vel|diesel|gasolina|etanol|oleo|óleo|lubrificante|posto)\b", "Combustível"),
    (r"\b(eletric(a|o)?|fiao|fiacao|fio|disjuntor|quadro|tomad(a|as)|interruptor(es)?|spot|led|cabeamento|cabo\s*eletric)\b", "Elétrico"),
    (r"\b(hidraul(ic|i|ica|ico)|hidrossanit(a|á)ri(o|a)|encanamento|encanar|cano(s)?|tubo(s)?|tubo\s*pex|pvc\b|joelho|te\b|luva\b|registro|torneira|ralo|caixa\s*d'?agua|caixa\s*d'?água|esgoto|bomba|sifao|sifão)\b", "Hidráulico"),
    (r"\b(revestimento(s)?|forro(\s*pvc)?|dry\s*wall|drywall|gesso(\s+acartonado)?)\b", "Revestimentos"),
    (r"\b(pintur(a|ar)|tinta(s)?|massa\s*corrida|selador|lixa|rolo|fita\s*crepe|spray|textura|zarcao|zarc[aã]o)\b", "Pintura"),
    (r"\b(estrutura|fundacao|funda[cç][aã]o|sapata|broca|estaca|viga|pilar|laje|baldrame|concreto|cimento|areia|brita|argamassa|reboco|graute|bloco|tijolo|alvenaria|vergalh|arma[cç][aã]o|forma|escoramento|perfil|perfis|metalon|metalons)\b", "Estrutura/Alvenaria"),
    (r"\b(porta(s)?|janela(s)?|janela\s+de\s+vidro|vidro|esquadria|aluminio|alum[ií]nio|fechadur(a|as)|fechadura(s)?|dobradi[cç]a(s)?|trinco|cadeado|temperado|kit\s*porta)\b", "Esquadrias/Vidro"),
    (r"\b(telha|calha|rufo|cumeeira|zinco|manta\s*t[eé]rmica|termoac(o|ô)stic)\b", "Cobertura"),
    (r"\b(acabamento|piso|porcelanato|ceramica|cerâmica|rodape|rodapé|azulejo|pastilha|rejunte|argamassa\s*colante|granito|marmore|mármore|bancada|silicone|vedacao|vedação|massa\s*acrilica|massa\s*acrílica|massa\s*fina|box\b|espelho|lou(c|ç)a(s)?|metais|vaso\s+sanitario|vaso\s+sanitário|cuba\b|pia\b|acabamento\s+de\s+tomada|espelho\s+de\s+tomada)\b", "Acabamento"),
    (r"\b(impermeabiliza|manta\s*asf[aá]ltica|vedacit|sika)\b", "Impermeabilização"),
    (r"\b(parafuso|parafusos|prego|pregos|bucha|buchas|rebite|rebites|arruela|arruelas|porca|porcas|solda|eletrodo|eletrodos|arame\s*solda|disco|discos|disco\s*corte|disco\s*flap|lixa|lixas|abra[cç]adeira|enforca\s*gato|hellermann|silicone|cola|super\s*bonder|espuma\s*expansiva|vedante)\b", "Consumíveis"),
    (r"\b(analgesico|analg[eé]sico|dipirona|ibuprofeno|paracetamol|remedio|rem[eé]dios?|pomada|curativo|gaze|esparadrapo|bandagem|antisseptico|antiss[eé]ptico|protetor\s*solar|luva|luvas|oculos|[óo]culos|oculos\s*prote[cç][aã]o|[óo]culos\s*prote[cç][aã]o|capacete|protetor\s*auricular|mascara|m[aá]scara|respirador|bota|botina)\b", "SST"),
    (r"\b(parafus(o|os)|broca(s)?|eletrodo(s)?|disco\s*corte|abracadeira|abra[cç]adeira|chumbador|rebite|arruela|porca)\b", "Ferragens/Consumíveis"),
    (r"\b(esmerilhadeira|serra\s*circular|lixadeira|parafusadeira|multimetro|trena)\b", "Ferramentas"),
    (r"\b(bobcat|compactador|gerador|betoneira|aluguel\s*equip|loca[cç][aã]o\s*equip|plataforma|guindaste)\b", "Equipamentos"),
    (r"\b(trafego|tr[aá]fego|ads|google|meta|facebook|instagram|impulsionamento|an[uú]ncio)\b", "Marketing"),
    (r"\b(aluguel|internet|energia|conta\s*de\s*luz|conta\s*de\s*agua|conta\s*de\s*água|telefone|contabilidade|escritorio|escritório)\b", "Custos Fixos"),
    (r"\b(taxa|emolumento|cartorio|cartório|crea|art|multa|juros|tarifa|banco|ted\b|boleto|iof)\b", "Taxas/Financeiro"),
    (r"\b(agua\s+(para\s+beber|de\s+beber)|agua\b|comida(s)?|refeic(a|ã)o|refei[cç][oõ]es|lanche(s)?|marmit(a|as)|quentinha(s)?|almo[cç]o(s)?|jantar(es)?|restaurante(s)?|cafe|café|refrigerante)\b", "Alimentação"),
]
DEFAULT_CATEGORY = "Outros"

PAYMENT_SYNONYMS = {
    "PIX": r"\bpix\b|\bfiz\s+um\s+pix\b|\bmandei\s+um\s+pix\b|\bchave\s+pix\b",
    "CREDITO": r"\bcr[eé]dito\b|\bno\s+cart[aã]o\s+de\s+cr[eé]dito\b|\bpassei\s+no\s+cr[eé]dito\b|\bpassei\s+no\s+cart[aã]o\b",
    "DEBITO": r"\bd[eé]bito\b|\bno\s+cart[aã]o\s+de\s+d[eé]bito\b|\bpassei\s+no\s+d[eé]bito\b",
    "DINHEIRO": r"\bdinheiro\b|\bem\s+esp[eé]cie\b|\bcash\b",
    "VALE": r"\bvale\b|\badiantamento\b",
}

QUERY_INTENT_RE = re.compile(
    r"\b("
    r"quanto\s+(eu\s+)?gastei|"
    r"quanto\s+foi\s+que\s+eu\s+gastei|"
    r"gastos?|despesas?|"
    r"relat[oó]rio|resumo|"
    r"me\s+mostra|mostra\s+pra\s+mim|me\s+manda|"
    r"lista|detalha|detalhar|itens?|lan[cç]amentos?|extrato|"
    r"quanto\s+(ja\s+)?recebi|"
    r"quanto\s+entrou|"
    r"receitas?|entradas?|"
    r"total\s+recebido|"
    r"saldo(\s+atual)?|"
    r"lucro|margem|dre|"
    r"ranking"
    r")\b",
    re.I
)

SALDO_INTENT_RE = re.compile(r"\b(saldo(\s+atual)?|balanc(o|co)|balanco)\b", re.I)
COMPANY_BALANCE_RE = re.compile(r"\b(saldo\s+da\s+empresa|saldo\s+geral|geral\s+da\s+empresa|empresa\s+no\s+geral|saldo\s+total)\b", re.I)
PROFIT_INTENT_RE = re.compile(r"\b(lucro|margem|dre)\b", re.I)
RANKING_INTENT_RE = re.compile(r"\b(ranking)\b", re.I)

# =====================================================================================
#                               PROCESSAMENTO ÚNICO (texto e áudio) - helpers de período
# =====================================================================================

def has_explicit_period(text: str) -> bool:
    low = _norm(text or "")
    date_re = re.compile(r"\b\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?\b")
    period_markers = [
        "este mes", "essa semana", "semana passada", "mes passado",
        "ultimos", "últimos", "hoje", "ontem", "ultima quinzena", "última quinzena",
        "janeiro", "fevereiro", "marco", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
    ]
    if date_re.search(low):
        return True
    return any(mk in low for mk in period_markers)


def all_time_period_label():
    today = date.today()
    start = date(1900, 1, 1).isoformat()
    end = (today + timedelta(days=1)).isoformat()
    return start, end, "todo o período"

# =====================================================================================
#                          FILTROS / INTENTS (CONSULTAS)
# =====================================================================================

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


def is_company_balance_request(text: str) -> bool:
    return bool(COMPANY_BALANCE_RE.search(text or ""))


EXPENSE_HINT_RE = re.compile(r"\b(despesa|despesas|gasto|gastei|paguei|comprei|saida|saída)\b", re.I)


def is_income_query(text: str):
    t = _norm(text or "")
    return bool(re.search(
        r"\b(entrou|entrada(s)?|recebi|receber|recebido|recebida|receita(s)?|pagaram|pagamento(s)?|quanto\s+(ja\s+)?recebi|quanto\s+entrou|total\s+recebido|pix\s+recebid[oa])\b",
        t,
        re.I
    ))


def detect_type_filter(text: str) -> str | None:
    t = _norm(text or "")
    if is_income_query(t):
        return "income"
    if EXPENSE_HINT_RE.search(t):
        return "expense"
    if "extrato" in t:
        return None
    if re.search(r"\b(lista|lan[cç]amentos?)\b", t):
        return None
    return "expense"


def is_report_intent(text: str):
    return bool(QUERY_INTENT_RE.search(text or ""))


def is_saldo_intent(text: str):
    return bool(SALDO_INTENT_RE.search(text or ""))


def is_summary_request(text: str):
    t = _norm(text or "")
    return bool(re.search(r"\b(resumo|relatorio|relatório)\b", t))


def is_list_request(text: str):
    t = _norm(text or "")
    return bool(re.search(r"\b(lista|detalha|detalhar|itens?|lan[cç]amentos?|extrato)\b", t))


def is_profit_request(text: str):
    return bool(PROFIT_INTENT_RE.search(text or ""))


def is_ranking_request(text: str):
    return bool(RANKING_INTENT_RE.search(text or ""))

# =====================================================================================
#                               CC STATE + pending + undo cache
# =====================================================================================

PENDING_BY_USER: dict[int, dict] = {}
LAST_ENTRY_BY_TG_USER: dict[int, dict] = {}


def _get_user_row(tg_user_id: int) -> dict | None:
    r = sb.table("users").select("id,role,is_active,account_id,last_cc").eq("tg_user_id", tg_user_id).limit(1).execute()
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


def _get_last_entry_id_from_db(tg_user_id: int) -> int | None:
    try:
        r = sb.table("users").select("last_entry_id").eq("tg_user_id", tg_user_id).limit(1).execute()
        rows = get_or_none(r) or []
        if rows and rows[0].get("last_entry_id") is not None:
            try:
                return int(rows[0]["last_entry_id"])
            except Exception:
                return None
    except Exception:
        pass
    return None


def _set_last_entry_id_to_db(tg_user_id: int, entry_id: int | None):
    try:
        payload = {
            "last_entry_id": entry_id,
            "last_entry_at": datetime.datetime.utcnow().isoformat()
        }
        sb.table("users").update(payload).eq("tg_user_id", tg_user_id).execute()
    except Exception:
        pass

# =====================================================================================
#                               CORREÇÕES (EDITAR ÚLTIMO LANÇAMENTO)
# =====================================================================================

def is_correction_intent(text: str) -> bool:
    t_norm = _norm(text or "").strip()

    if re.fullmatch(r"(em\s+)?\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?", t_norm):
        if parse_date_pt(text) and money_from_text(text) is None:
            return True

    return bool(re.search(
        r"\b(corrig(e|ir|e ai)|ajusta(r|a)|editar|edita|troca(r)?|muda(r)?|mudar\b|nao\s+e|não\s+é|na\s+verdade|o\s+certo\s+e|era\s+.*\s+mas\s+e)\b",
        t_norm
    ))


def extract_correction_targets(text: str) -> dict:
    t = _norm(text or "")

    found_cats = []
    for pat, cat in CATEGORY_RULES:
        if re.search(pat, t):
            found_cats.append(cat)
    cat = found_cats[-1] if found_cats else None

    cc = guess_cc_strict_for_correction(text)
    dtx = parse_date_pt(text)
    paid_via = guess_payment(text)

    typ = None
    if re.search(r"\b(receita|entrada|entrou|recebi|pix\s+recebid[oa])\b", t):
        typ = "income"
    if re.search(r"\b(despesa|gasto|paguei|pago|comprei|sa[ií]da)\b", t):
        typ = "expense" if typ is None else typ

    new_amount = money_from_text(text)

    return {
        "category_name": cat,
        "cc_code": cc,
        "entry_date": dtx,
        "paid_via": paid_via,
        "type": typ,
        "amount": new_amount
    }


def _resolve_last_entry_for_user(account_id: str, user_id: int, tg_uid: int) -> dict | None:
    db_id = _get_last_entry_id_from_db(tg_uid)
    if db_id:
        try:
            r = sb.table("entries").select(
                "id,account_id,created_by,amount,type,entry_date,category_id,cost_center_id,paid_via,description"
            ).eq("id", db_id).limit(1).execute()
            rows = get_or_none(r) or []
            if rows:
                row = rows[0]
                if row.get("account_id") == account_id and row.get("created_by") == user_id:
                    return row
        except Exception:
            pass

    meta = LAST_ENTRY_BY_TG_USER.get(tg_uid) or {}
    if meta.get("id") is not None:
        try:
            eid = int(meta["id"])
            r = sb.table("entries").select(
                "id,account_id,created_by,amount,type,entry_date,category_id,cost_center_id,paid_via,description"
            ).eq("id", eid).limit(1).execute()
            rows = get_or_none(r) or []
            if rows:
                row = rows[0]
                if row.get("account_id") == account_id and row.get("created_by") == user_id:
                    return row
        except Exception:
            pass

    try:
        r = sb.table("entries").select(
            "id,account_id,created_by,amount,type,entry_date,category_id,cost_center_id,paid_via,description"
        ).eq("account_id", account_id).eq("created_by", user_id).order("created_at", desc=True).limit(1).execute()
        rows = get_or_none(r) or []
        return rows[0] if rows else None
    except Exception:
        return None


async def try_apply_correction(update: Update, text: str) -> bool:
    tg_uid = update.effective_user.id
    user_row = _get_user_row(tg_uid)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return True

    account_id = user_row.get("account_id") or get_default_account_id()
    user_id = user_row["id"]

    last_row = _resolve_last_entry_for_user(account_id, user_id, tg_uid)
    if not last_row:
        await update.message.reply_text("⚠️ Não encontrei um lançamento recente pra corrigir.")
        return True

    targets = extract_correction_targets(text)
    if not any([
        targets.get("category_name"),
        targets.get("cc_code"),
        targets.get("entry_date"),
        targets.get("paid_via"),
        targets.get("type"),
        targets.get("amount") is not None
    ]):
        return False

    upd_payload = {}

    if targets.get("category_name"):
        cat_id = ensure_category_id(account_id, targets["category_name"])
        if cat_id:
            upd_payload["category_id"] = cat_id

    if targets.get("cc_code"):
        cc_id = ensure_cost_center_id(account_id, targets["cc_code"])
        if cc_id:
            upd_payload["cost_center_id"] = cc_id
            _set_last_cc(tg_uid, targets["cc_code"])

    if targets.get("entry_date"):
        upd_payload["entry_date"] = targets["entry_date"]

    if targets.get("paid_via"):
        upd_payload["paid_via"] = targets["paid_via"]

    if targets.get("type") in ("income", "expense"):
        upd_payload["type"] = targets["type"]

    if targets.get("amount") is not None:
        upd_payload["amount"] = float(targets["amount"])

    if not upd_payload:
        return False

    try:
        sb.table("entries").update(upd_payload).eq("id", last_row["id"]).execute()
    except Exception as e:
        await update.message.reply_text(f"💥 Não consegui corrigir agora: {type(e).__name__}: {e}")
        return True

    changed = []
    if targets.get("amount") is not None:
        changed.append(f"Valor → {moeda_fmt(float(targets['amount']))}")
    if targets.get("category_name"):
        changed.append(f"Categoria → {targets['category_name']}")
    if targets.get("cc_code"):
        changed.append(f"CC → {targets['cc_code']}")
    if targets.get("entry_date"):
        changed.append(f"Data → {data_fmt_out(targets['entry_date'])}")
    if targets.get("paid_via"):
        changed.append(f"Pagamento → {targets['paid_via']}")
    if targets.get("type"):
        changed.append(f"Tipo → {'Receita' if targets['type'] == 'income' else 'Despesa'}")

    await update.message.reply_text("✅ Ajustado: " + " | ".join(changed))

    try:
        _set_last_entry_id_to_db(tg_uid, int(last_row["id"]))
    except Exception:
        pass

    return True

# =====================================================================================
#                               PERSISTÊNCIA (LANÇAMENTO)
# =====================================================================================

def save_entry(tg_user_id: int, txt: str, force_cc: str | None = None):
    try:
        low = _norm(txt)

        amount = money_from_text(txt)
        if amount is None:
            return False, "Não achei o valor. Ex.: 'paguei 200 no eletricista'."

        if re.search(r"\b(receb(i|ido|ida|imento)|receita|entrada|entrou|vendi|pagaram|pagou\s+pra\s+mim|pix\s+recebid[oa])\b", low):
            etype = "income"
        else:
            etype = "expense"

        user_row = _get_user_row(tg_user_id)
        if not user_row or not user_row.get("is_active"):
            return False, "Usuário não autorizado. Use /start e peça autorização."

        user_id = user_row["id"]
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

        payload = {
            "account_id": account_id,
            "entry_date": entry_date,
            "type": etype,
            "amount": amount,
            "description": txt,
            "category_id": cat_id,
            "cost_center_id": cc_id,
            "paid_via": paid_via,
            "status": "approved",
            "created_by": user_id,
        }

        created_row = None

        try:
            ins_res = sb.table("entries").insert(payload).execute()
            created = get_or_none(ins_res) or []
            created_row = created[0] if created else None
        except Exception:
            payload.pop("paid_via", None)
            ins_res = sb.table("entries").insert(payload).execute()
            created = get_or_none(ins_res) or []
            created_row = created[0] if created else None

        if cc_code:
            _set_last_cc(tg_user_id, cc_code)

        if created_row and isinstance(created_row, dict) and created_row.get("id") is not None:
            LAST_ENTRY_BY_TG_USER[tg_user_id] = {
                "id": created_row["id"],
                "account_id": account_id,
                "created_by": user_id,
                "amount": amount,
                "type": etype,
                "entry_date": entry_date,
                "cc": cc_code,
                "category": cat_name,
            }
        else:
            LAST_ENTRY_BY_TG_USER[tg_user_id] = {
                "id": None,
                "account_id": account_id,
                "created_by": user_id,
                "amount": amount,
                "type": etype,
                "entry_date": entry_date,
                "cc": cc_code,
                "category": cat_name,
            }

        if created_row and isinstance(created_row, dict) and created_row.get("id") is not None:
            try:
                _set_last_entry_id_to_db(tg_user_id, int(created_row["id"]))
            except Exception:
                pass

        return True, {
            "amount": amount,
            "type": etype,
            "category": cat_name,
            "cc": cc_code,
            "paid_via": paid_via,
            "entry_date": entry_date,
            "used_last_cc": used_last_cc
        }

    except Exception as e:
        return False, f"Erro interno no save_entry: {type(e).__name__}: {e}"

# =====================================================================================
#                               CONSULTAS / RELATÓRIOS
# =====================================================================================

async def cmd_resumo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_row = _get_user_row(update.effective_user.id)
        if not user_row or not user_row.get("is_active"):
            await update.message.reply_text("Usuário não autorizado.")
            return

        account_id = user_row.get("account_id") or get_default_account_id()

        today = datetime.date.today()
        start = _first_day_of_week(today).isoformat()
        end = (_first_day_of_week(today) + timedelta(days=7)).isoformat()
        label = "essa semana"

        resp = sb.table("entries").select(
            "amount,category_id,cost_center_id,type"
        ).eq("account_id", account_id).gte("entry_date", start).lt("entry_date", end).eq("type", "expense").execute()

        rows = get_or_none(resp) or []
        if not rows:
            await update.message.reply_text("📊 Nenhum gasto registrado essa semana.")
            return

        cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
        ccs_rows = get_or_none(sb.table("cost_centers").select("id,code").eq("account_id", account_id).execute()) or []
        cats = {r["id"]: r["name"] for r in cats_rows}
        ccs = {r["id"]: r["code"] for r in ccs_rows}

        by_cat = defaultdict(float)
        by_cc = defaultdict(float)
        total = 0.0
        for r in rows:
            v = float(r["amount"])
            total += v
            by_cat[cats.get(r["category_id"], "Sem categoria")] += v
            by_cc[ccs.get(r["cost_center_id"], "Sem CC")] += v

        def top_fmt(d, limit=5):
            items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:limit]
            return "\n".join([f"• {k}: {moeda_fmt(v)}" for k, v in items]) or "• (sem lançamentos)"

        msg = (
            f"📊 Resumo semanal — despesas\n"
            f"Período: {label}\n"
            f"——————————————\n"
            f"Total gasto: {moeda_fmt(total)}\n\n"
            f"Top categorias:\n{top_fmt(by_cat)}\n\n"
            f"Top centros de custo:\n{top_fmt(by_cc)}"
        )
        await _send_long(update, msg)

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no /resumo: {type(e).__name__}: {e}")


async def cmd_desfazer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tg_uid = update.effective_user.id

        if tg_uid in PENDING_BY_USER:
            PENDING_BY_USER.pop(tg_uid, None)
            await update.message.reply_text("↩️ Beleza. Pendência cancelada (não lancei nada).")
            return

        user_row = _get_user_row(tg_uid)
        if not user_row or not user_row.get("is_active"):
            await update.message.reply_text("Usuário não autorizado.")
            return

        account_id = user_row.get("account_id") or get_default_account_id()
        user_id = user_row["id"]

        entry_id = None
        meta = LAST_ENTRY_BY_TG_USER.get(tg_uid) or {}

        db_last_entry_id = _get_last_entry_id_from_db(tg_uid)
        if db_last_entry_id:
            try:
                r = sb.table("entries").select("id,amount,type,entry_date,account_id,created_by").eq("id", db_last_entry_id).limit(1).execute()
                rows = get_or_none(r) or []
                if rows:
                    row = rows[0]
                    if row.get("account_id") == account_id and row.get("created_by") == user_id:
                        entry_id = row["id"]
                        meta = {**meta, **row}
            except Exception:
                pass

        if not entry_id:
            if meta and meta.get("id") is not None and meta.get("account_id") == account_id and meta.get("created_by") == user_id:
                entry_id = meta["id"]

        if not entry_id:
            r = sb.table("entries").select("id,amount,type,entry_date").eq("account_id", account_id).eq("created_by", user_id).order("created_at", desc=True).limit(1).execute()
            rows = get_or_none(r) or []
            if rows:
                entry_id = rows[0]["id"]
                meta = {**meta, **rows[0]}

        if not entry_id:
            await update.message.reply_text("↩️ Não encontrei nenhum lançamento recente pra desfazer.")
            return

        sb.table("entries").delete().eq("id", entry_id).execute()

        LAST_ENTRY_BY_TG_USER.pop(tg_uid, None)
        _set_last_entry_id_to_db(tg_uid, None)

        typ = meta.get("type")
        icon = entry_emoji(typ)

        msg = f"{icon} Desfeito: "
        msg += "última receita" if typ == "income" else "última despesa" if typ == "expense" else "último lançamento"

        extras = []
        amt = meta.get("amount")
        if amt is not None:
            try:
                extras.append(moeda_fmt(float(amt)))
            except Exception:
                pass

        dt = meta.get("entry_date")
        if dt:
            extras.append(f"🗓️ {data_fmt_out(dt)}")

        if extras:
            msg += "\n" + " • ".join(extras)

        await update.message.reply_text(msg)

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no /desfazer: {type(e).__name__}: {e}")


def _fmt_ranked(d: dict[str, float], limit: int | None = None) -> str:
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    if limit:
        items = items[:limit]
    return "\n".join([f"• {k}: {moeda_fmt(v)}" for k, v in items]) or "• (sem lançamentos)"


async def run_query_and_reply(update: Update, text: str):
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return
    account_id = user_row.get("account_id") or get_default_account_id()

    cc_code = guess_cc_filter(text)
    paid = guess_paid_filter(text)
    cat = guess_category_filter(text)

    if cc_code and not has_explicit_period(text):
        start, end, label = all_time_period_label()
    else:
        start, end, label = parse_period_pt(text)

    typ = detect_type_filter(text)

    q = sb.table("entries").select("amount,category_id,cost_center_id,paid_via,type,entry_date").eq("account_id", account_id).gte("entry_date", start).lt("entry_date", end)

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

    if typ in ("income", "expense"):
        q = q.eq("type", typ)

    rows = get_or_none(q.execute()) or []

    receitas = sum(float(r["amount"]) for r in rows if r.get("type") == "income")
    despesas = sum(float(r["amount"]) for r in rows if r.get("type") == "expense")
    saldo = receitas - despesas

    filtros = []
    if cc_code:
        filtros.append(cc_code)
    if cat:
        filtros.append(cat)
    if paid:
        filtros.append(paid.title())
    filtros_txt = f" | Filtros: {', '.join(filtros)}" if filtros else ""

    if typ == "income":
        msg = f"📊 Total de receitas em {label}{filtros_txt}:\n{moeda_fmt(receitas)}"
    elif typ == "expense":
        msg = f"📊 Total de despesas em {label}{filtros_txt}:\n{moeda_fmt(despesas)}"
    else:
        msg = (
            f"📊 Totais em {label}{filtros_txt}\n"
            f"Receitas: {moeda_fmt(receitas)}\n"
            f"Despesas: {moeda_fmt(despesas)}\n"
            f"——————————————\n"
            f"Saldo: {moeda_fmt(saldo)}"
        )

    await update.message.reply_text(msg)


async def run_list_entries_and_reply(update: Update, text: str):
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return
    account_id = user_row.get("account_id") or get_default_account_id()

    cc_code = guess_cc_filter(text)
    cat = guess_category_filter(text)

    if cc_code and not has_explicit_period(text):
        start, end, label = all_time_period_label()
    else:
        start, end, label = parse_period_pt(text)

    typ = detect_type_filter(text)

    low = _norm(text or "")
    limit = 2000 if re.search(r"\b(completo|tudo|inteiro)\b", low) else 200

    q = sb.table("entries").select("amount,type,entry_date,description,category_id,cost_center_id").eq("account_id", account_id).gte("entry_date", start).lt("entry_date", end).order("entry_date", desc=True).limit(limit)

    if cc_code:
        cc = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", cc_code).limit(1).execute()
        ccd = get_or_none(cc) or []
        if ccd:
            q = q.eq("cost_center_id", ccd[0]["id"])

    if cat:
        c = sb.table("categories").select("id").eq("account_id", account_id).eq("name", cat).limit(1).execute()
        cd = get_or_none(c) or []
        if cd:
            q = q.eq("category_id", cd[0]["id"])

    if typ in ("income", "expense"):
        q = q.eq("type", typ)

    rows = get_or_none(q.execute()) or []
    if not rows:
        await update.message.reply_text(f"📄 Sem lançamentos em {label}" + (f" para {cc_code}" if cc_code else "") + ".")
        return

    cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
    ccs_rows = get_or_none(sb.table("cost_centers").select("id,code").eq("account_id", account_id).execute()) or []
    cats = {r["id"]: r["name"] for r in cats_rows}
    ccs = {r["id"]: r["code"] for r in ccs_rows}

    receitas = sum(float(r["amount"]) for r in rows if r.get("type") == "income")
    despesas = sum(float(r["amount"]) for r in rows if r.get("type") == "expense")
    saldo = receitas - despesas

    by_cat_exp = defaultdict(float)
    by_cat_inc = defaultdict(float)

    lines = []
    for r in rows:
        v = float(r["amount"])
        dt = data_fmt_out(r.get("entry_date"))
        cat_name = cats.get(r.get("category_id"), "Sem categoria")
        cc_name = ccs.get(r.get("cost_center_id"), "Sem CC")
        desc = _clip(r.get("description") or "", 60)

        if r.get("type") == "income":
            by_cat_inc[cat_name] += v
            icon = "✅"
        else:
            by_cat_exp[cat_name] += v
            icon = "⛔"

        lines.append(f"• {dt} {icon} {moeda_fmt(v)} — {cat_name} — {cc_name}\n  {_clip(desc, 90)}")

    header = f"📄 Extrato (lista) — {label}"
    if cc_code:
        header += f" | {cc_code}"
    if cat:
        header += f" | {cat}"

    note = ""
    if len(rows) >= limit:
        note = f"\n⚠️ Mostrando até {limit} itens. Se quiser garantir tudo: escreva “extrato completo {cc_code or ''} {label}”."

    msg = (
        header + "\n"
        "——————————————\n"
        f"Receitas: {moeda_fmt(receitas)}\n"
        f"Despesas: {moeda_fmt(despesas)}\n"
        f"Saldo: {moeda_fmt(saldo)}\n\n"
        f"Resumo por categoria (despesas):\n{_fmt_ranked(by_cat_exp, limit=10)}\n\n"
        f"Resumo por categoria (receitas):\n{_fmt_ranked(by_cat_inc, limit=10)}\n"
        + note +
        "\n\nLançamentos:\n" + "\n".join(lines)
    )

    await _send_long(update, msg)


async def run_balance_and_reply(update: Update, text: str):
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return
    account_id = user_row.get("account_id") or get_default_account_id()

    cc_code = None if is_company_balance_request(text) else guess_cc_filter(text)

    if cc_code and not has_explicit_period(text):
        start, end, label = all_time_period_label()
    else:
        start, end, label = parse_period_pt(text)

    base = sb.table("entries").select("amount,type,cost_center_id,entry_date").eq("account_id", account_id).gte("entry_date", start).lt("entry_date", end)

    if cc_code:
        cc = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", cc_code).limit(1).execute()
        ccd = get_or_none(cc) or []
        if ccd:
            base = base.eq("cost_center_id", ccd[0]["id"])

    rows = get_or_none(base.execute()) or []
    receitas = sum(float(r["amount"]) for r in rows if r.get("type") == "income")
    despesas = sum(float(r["amount"]) for r in rows if r.get("type") == "expense")
    saldo = receitas - despesas

    filtro_txt = " | Empresa (geral)" if is_company_balance_request(text) else (f" | {cc_code}" if cc_code else "")

    msg = (
        f"💰 Saldo em {label}{filtro_txt}\n"
        f"Receitas: {moeda_fmt(receitas)}\n"
        f"Despesas: {moeda_fmt(despesas)}\n"
        f"——————————————\n"
        f"Saldo: {moeda_fmt(saldo)}"
    )
    await update.message.reply_text(msg)


async def run_profit_and_reply(update: Update, text: str):
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return

    account_id = user_row.get("account_id") or get_default_account_id()

    cc_code = guess_cc_filter(text)
    if not cc_code:
        await update.message.reply_text("Informe a obra ou container. Ex: lucro do container do Thiago")
        return

    if cc_code and not has_explicit_period(text):
        start, end, label = all_time_period_label()
    else:
        start, end, label = parse_period_pt(text)

    cc_res = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", cc_code).limit(1).execute()
    cc_rows = get_or_none(cc_res) or []
    if not cc_rows:
        await update.message.reply_text(f"Sem centro de custo cadastrado para {cc_code}")
        return

    cc_id = cc_rows[0]["id"]

    rows = get_or_none(
        sb.table("entries").select("amount,type,category_id").eq("account_id", account_id).eq("cost_center_id", cc_id).gte("entry_date", start).lt("entry_date", end).execute()
    ) or []

    if not rows:
        await update.message.reply_text(f"Sem lançamentos para {cc_code} em {label}.")
        return

    cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
    cats_map = {r["id"]: r["name"] for r in cats_rows}

    receitas = 0.0
    despesas = 0.0
    cats = {}

    for r in rows:
        val = float(r["amount"])
        if r["type"] == "income":
            receitas += val
        else:
            despesas += val
            cat = cats_map.get(r.get("category_id"), "Outros")
            cats[cat] = cats.get(cat, 0.0) + val

    lucro = receitas - despesas
    margem = (lucro / receitas * 100.0) if receitas > 0 else 0.0

    msg = f"📊 DRE — {cc_code}\n\n"
    msg += "Receita total\n"
    msg += f"{moeda_fmt(receitas)}\n\n"
    msg += "Despesas\n"
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        msg += f"{k}: {moeda_fmt(v)}\n"
    msg += "\nTotal despesas\n"
    msg += f"{moeda_fmt(despesas)}\n\n"
    msg += "💰 Lucro\n"
    msg += f"{moeda_fmt(lucro)}\n\n"
    msg += "Margem\n"
    msg += f"{margem:.1f}%\n\n"
    msg += "📌 Ranking de gastos\n"

    pos = 1
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        msg += f"{pos}️⃣ {k}\n{moeda_fmt(v)}\n\n"
        pos += 1
        if pos > 10:
            break

    await _send_long(update, msg.strip())


async def run_ranking_and_reply(update: Update, text: str):
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return

    account_id = user_row.get("account_id") or get_default_account_id()

    if has_explicit_period(text):
        start, end, label = parse_period_pt(text)
    else:
        start, end, label = all_time_period_label()

    rows = get_or_none(
        sb.table("entries").select("amount,category_id").eq("account_id", account_id).eq("type", "expense").gte("entry_date", start).lt("entry_date", end).execute()
    ) or []

    if not rows:
        await update.message.reply_text("Sem despesas registradas.")
        return

    cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
    cats_map = {r["id"]: r["name"] for r in cats_rows}

    cats = {}
    for r in rows:
        cat = cats_map.get(r.get("category_id"), "Outros")
        cats[cat] = cats.get(cat, 0.0) + float(r["amount"])

    ordered = sorted(cats.items(), key=lambda x: -x[1])

    msg = f"📊 Ranking de gastos — {label}\n\n"
    pos = 1
    for cat, val in ordered:
        msg += f"{pos}️⃣ {cat}\n{moeda_fmt(val)}\n\n"
        pos += 1
        if pos > 10:
            break

    await _send_long(update, msg.strip())


async def run_cc_full_summary(update: Update, text: str):
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return
    account_id = user_row.get("account_id") or get_default_account_id()

    cc_code = guess_cc_filter(text)
    if not cc_code:
        await update.message.reply_text("Me diz qual centro de custo. Ex: 'resumo da obra do Rodrigo' ou 'resumo do container do Thiago'.")
        return

    if not has_explicit_period(text):
        start, end, label = all_time_period_label()
    else:
        start, end, label = parse_period_pt(text)

    cc = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", cc_code).limit(1).execute()
    ccd = get_or_none(cc) or []
    if not ccd:
        await update.message.reply_text(f"Não achei esse centro de custo: {cc_code}")
        return
    cc_id = ccd[0]["id"]

    rows = get_or_none(
        sb.table("entries").select("amount,type,category_id").eq("account_id", account_id).eq("cost_center_id", cc_id).gte("entry_date", start).lt("entry_date", end).execute()
    ) or []

    if not rows:
        await update.message.reply_text(f"📊 Sem lançamentos em {label} para {cc_code}.")
        return

    cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
    cats = {r["id"]: r["name"] for r in cats_rows}

    total_in = 0.0
    total_out = 0.0
    by_cat_exp = defaultdict(float)
    by_cat_inc = defaultdict(float)

    for r in rows:
        v = float(r["amount"])
        cat_name = cats.get(r.get("category_id"), "Sem categoria")
        if r.get("type") == "income":
            total_in += v
            by_cat_inc[cat_name] += v
        else:
            total_out += v
            by_cat_exp[cat_name] += v

    saldo = total_in - total_out

    msg = (
        f"📊 Resumo — {cc_code}\n"
        f"Período: {label}\n"
        f"——————————————\n"
        f"Receitas: {moeda_fmt(total_in)}\n"
        f"Despesas: {moeda_fmt(total_out)}\n"
        f"Saldo: {moeda_fmt(saldo)}\n\n"
        f"Despesas por categoria:\n{_fmt_ranked(by_cat_exp, limit=10)}\n\n"
        f"Receitas por categoria:\n{_fmt_ranked(by_cat_inc, limit=10)}"
    )
    await _send_long(update, msg)


async def run_cc_extrato(update: Update, cc_code: str, period_text: str | None = None):
    user_row = _get_user_row(update.effective_user.id)
    if not user_row or not user_row.get("is_active"):
        await update.message.reply_text("Usuário não autorizado.")
        return
    account_id = user_row.get("account_id") or get_default_account_id()

    base_text = period_text or ""
    if not base_text or not has_explicit_period(base_text):
        start, end, label = all_time_period_label()
    else:
        start, end, label = parse_period_pt(base_text)

    cc = sb.table("cost_centers").select("id").eq("account_id", account_id).eq("code", cc_code).limit(1).execute()
    ccd = get_or_none(cc) or []
    if not ccd:
        await update.message.reply_text(f"Não achei esse centro de custo: {cc_code}")
        return
    cc_id = ccd[0]["id"]

    rows = get_or_none(
        sb.table("entries").select("amount,type,category_id,entry_date,description").eq("account_id", account_id).eq("cost_center_id", cc_id).gte("entry_date", start).lt("entry_date", end).order("entry_date", desc=True).limit(2000).execute()
    ) or []

    if not rows:
        await update.message.reply_text(f"📄 Sem lançamentos em {label} para {cc_code}.")
        return

    cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
    cats = {r["id"]: r["name"] for r in cats_rows}

    total_in = 0.0
    total_out = 0.0
    by_cat_exp = defaultdict(float)
    by_cat_inc = defaultdict(float)

    for r in rows:
        v = float(r["amount"])
        cat_name = cats.get(r.get("category_id"), "Sem categoria")
        if r.get("type") == "income":
            total_in += v
            by_cat_inc[cat_name] += v
        else:
            total_out += v
            by_cat_exp[cat_name] += v

    saldo = total_in - total_out

    last_lines = []
    for r in rows[:12]:
        icon = entry_emoji(r.get("type"))
        dt = data_fmt_out(r.get("entry_date"))
        cat_name = cats.get(r.get("category_id"), "Sem categoria")
        last_lines.append(f"• {dt} {icon} {moeda_fmt(float(r['amount']))} — {cat_name}\n  {_clip(r.get('description') or '', 90)}")

    msg = (
        f"📒 Extrato — {cc_code}\n"
        f"Período: {label}\n"
        f"——————————————\n"
        f"Receitas: {moeda_fmt(total_in)}\n"
        f"Despesas: {moeda_fmt(total_out)}\n"
        f"Saldo: {moeda_fmt(saldo)}\n\n"
        f"Despesas por categoria:\n{_fmt_ranked(by_cat_exp, limit=10)}\n\n"
        f"Receitas por categoria:\n{_fmt_ranked(by_cat_inc, limit=10)}\n\n"
        f"Últimos lançamentos:\n" + "\n".join(last_lines)
    )
    await _send_long(update, msg)

# =====================================================================================
#                               PROCESSAMENTO ÚNICO (texto e áudio)
# =====================================================================================

async def process_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    try:
        uid = update.effective_user.id
        user_text = (user_text or "").strip()
        if not user_text:
            return

        user_row = _get_user_row(uid)

        if uid in PENDING_BY_USER:
            if not user_row or not user_row.get("is_active"):
                PENDING_BY_USER.pop(uid, None)
                await update.message.reply_text("⚠️ Usuário não autorizado. Use /start e peça autorização.")
                return

            cc = guess_cc_from_reply(user_text)
            if cc:
                pending = PENDING_BY_USER.pop(uid)
                ok, res = save_entry(uid, pending["txt"], force_cc=cc)
                if ok:
                    r = res
                    etype = r.get("type")
                    icon = entry_emoji(etype)
                    label = entry_label(etype)

                    extras = []
                    if r.get("paid_via"):
                        extras.append(f"💳 {r['paid_via']}")
                    if r.get("entry_date"):
                        extras.append(f"🗓️ {data_fmt_out(r['entry_date'])}")
                    tail = ("\n" + " • ".join(extras)) if extras else ""

                    await update.message.reply_text(
                        f"{icon} {label}: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc']}{tail}"
                    )
                else:
                    await update.message.reply_text(f"⚠️ {res}")
                return
            else:
                await update.message.reply_text("Não entendi o centro de custo. Ex: Bloco A, Sede, obra do Rodrigo, container do Thiago.")
                return

        if is_correction_intent(user_text):
            applied = await try_apply_correction(update, user_text)
            if applied:
                return

        low_text = _norm(user_text)

        if "lucro" in low_text or "margem" in low_text or "dre" in low_text:
            await run_profit_and_reply(update, user_text)
            return

        if "ranking" in low_text:
            await run_ranking_and_reply(update, user_text)
            return

        if is_list_request(user_text) and (is_report_intent(user_text) or guess_cc_filter(user_text) or is_income_query(user_text) or "extrato" in _norm(user_text)):
            await run_list_entries_and_reply(update, user_text)
            return

        if is_saldo_intent(user_text):
            await run_balance_and_reply(update, user_text)
            return

        if is_summary_request(user_text) and guess_cc_filter(user_text):
            await run_cc_full_summary(update, user_text)
            return

        if is_report_intent(user_text):
            await run_query_and_reply(update, user_text)
            return

        cc_only = guess_cc(user_text)
        if cc_only and money_from_text(user_text) is None and not is_report_intent(user_text) and not is_saldo_intent(user_text):
            if not user_row or not user_row.get("is_active"):
                await update.message.reply_text("⚠️ Usuário não autorizado. Use /start e peça autorização.")
                return

            _set_last_cc(uid, cc_only)
            await update.message.reply_text(f"✅ Centro de custo atual definido:\n{cc_only}")
            return

        cc_in_text = guess_cc(user_text)
        last_cc = _get_last_cc(uid)

        if (not cc_in_text) and (not last_cc) and money_from_text(user_text) is not None:
            if not user_row or not user_row.get("is_active"):
                await update.message.reply_text("⚠️ Usuário não autorizado. Use /start e peça autorização.")
                return

            PENDING_BY_USER[uid] = {"txt": user_text}
            await update.message.reply_text(
                "Beleza. Só me diz qual centro de custo pra eu lançar certinho.\n"
                "Ex: Bloco A, Sede/Administrativo, obra do Rodrigo, container do Thiago.\n\n"
                "Dica: define o CC do dia com /obra Rodrigo (ou 'bloco A', ou 'container Thiago')."
            )
            return

        ok, res = save_entry(uid, user_text)
        if ok:
            r = res
            etype = r.get("type")
            icon = entry_emoji(etype)
            label = entry_label(etype)

            extras = []
            if r.get("entry_date"):
                extras.append(f"🗓️ {data_fmt_out(r['entry_date'])}")
            if r.get("paid_via"):
                extras.append(f"💳 {r['paid_via']}")
            if r.get("used_last_cc") and r.get("cc"):
                extras.append(f"📌 CC assumido: {r['cc']}")
            tail = ("\n" + " • ".join(extras)) if extras else ""

            hint = ""
            if r.get("used_last_cc") and r.get("cc"):
                hint = "\nSe não for esse CC, manda: Bloco A / Sede / obra do <nome> / container do <nome> (que eu ajusto pro próximo)."

            await update.message.reply_text(
                f"{icon} {label}: {moeda_fmt(r['amount'])} • {r['category']} • {r['cc'] or 'Sem CC'}{tail}{hint}"
            )
        else:
            await update.message.reply_text(
                f"⚠️ {res}\n\n"
                "Exemplos:\n"
                "• paguei 200 no eletricista (pix) bloco A\n"
                "• pix recebido 1554,21 obra do Rodrigo\n"
                "• recebi mil do container do Thiago\n"
                "• paguei mil pra transportar o container da Ellen de munk\n"
                "• saldo do container da Ellen em fevereiro\n"
                "• saldo do container do Thiago\n"
                "• extrato do container do Thiago completo\n"
                "• saldo da empresa este mês\n"
                "• resumo do container do Thiago\n"
                "• lucro do container do Thiago\n"
                "• margem da obra do Rodrigo\n"
                "• dre do container da Ellen\n"
                "• ranking de gastos\n"
                "• ranking de gastos em fevereiro\n"
                "• me dá uma lista de tudo que eu gastei na obra do João em fevereiro\n"
                "• quanto já recebi na obra do João?\n"
                "• /extrato_obra João fevereiro\n"
                "• corrige, era 150, é 200\n"
                "• mudar a categoria para estrutura\n"
                "• /ajuda\n"
            )

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no processamento: {type(e).__name__}: {e}")

# =====================================================================================
#                               TELEGRAM HANDLERS
# =====================================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        u = update.effective_user
        default_account_id = get_default_account_id()

        exist = sb.table("users").select("*").eq("tg_user_id", u.id).limit(1).execute()
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
            upd = {
                "name": data[0].get("name") or u.full_name
            }

            if not data[0].get("account_id"):
                upd["account_id"] = default_account_id

            if "last_cc" not in data[0]:
                upd["last_cc"] = None

            if upd:
                sb.table("users").update(upd).eq("tg_user_id", u.id).execute()

        await update.message.reply_text(
            f"Fala, {u.first_name}! Eu sou o Boris.\n"
            f"Teu Telegram user id é: {u.id}\n"
            f"Pede pro owner te autorizar com /autorizar {u.id}\n"
            f"Comandos: /ajuda"
        )

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no /start: {type(e).__name__}: {e}")


async def cmd_autorizar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    you = _get_user_row(u.id)

    if not you or you.get("role") != "owner" or not you.get("is_active"):
        await update.message.reply_text("Somente o owner pode autorizar usuários.")
        return

    if len(context.args) == 0:
        await update.message.reply_text("Uso: /autorizar <tg_user_id>")
        return

    target = int(context.args[0])
    owner_account_id = you.get("account_id") or get_default_account_id()

    sb.table("users").upsert({
        "tg_user_id": target,
        "role": "buyer",
        "is_active": True,
        "name": "",
        "account_id": owner_account_id
    }).execute()

    await update.message.reply_text(f"Usuário {target} autorizado ✅")


async def cmd_obra(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = " ".join(context.args).strip()
    if not name:
        await update.message.reply_text("Uso: /obra <nome>. Ex: /obra Rodrigo")
        return
    cc = f"OBRA_{_slugify_name(name)}"
    _set_last_cc(update.effective_user.id, cc)
    await update.message.reply_text(f"✅ Centro de custo atual definido:\n{cc}")


async def cmd_despesa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args).strip()
    if not txt:
        await update.message.reply_text("Uso: /despesa <texto>. Ex: /despesa 200 eletricista pix bloco A")
        return
    await process_user_text(update, context, txt)


async def cmd_receita(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args).strip()
    if not txt:
        await update.message.reply_text("Uso: /receita <texto>. Ex: /receita 1200 pagamento Joana pix sede")
        return
    await process_user_text(update, context, "recebi " + txt)


async def cmd_saldo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = " ".join(context.args).strip()
    await run_balance_and_reply(update, txt or "este mês")


async def cmd_relatorio(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            "amount,category_id,cost_center_id,type,entry_date"
        ).eq("account_id", account_id).gte("entry_date", month_start).lt("entry_date", month_end).eq("type", "expense").execute()

        rows = get_or_none(resp) or []

        cats_rows = get_or_none(sb.table("categories").select("id,name").eq("account_id", account_id).execute()) or []
        ccs_rows = get_or_none(sb.table("cost_centers").select("id,code").eq("account_id", account_id).execute()) or []
        cats = {r["id"]: r["name"] for r in cats_rows}
        ccs = {r["id"]: r["code"] for r in ccs_rows}

        by_cat = defaultdict(float)
        by_cc = defaultdict(float)
        total = 0.0
        for r in rows:
            v = float(r["amount"])
            total += v
            by_cat[cats.get(r["category_id"], "Sem categoria")] += v
            by_cc[ccs.get(r["cost_center_id"], "Sem CC")] += v

        msg = (
            f"📊 Resumo do mês (despesas)\n"
            f"Total: {moeda_fmt(total)}\n\n"
            f"Por categoria:\n{_fmt_ranked(by_cat)}\n\n"
            f"Por centro de custo:\n{_fmt_ranked(by_cc)}"
        )
        await _send_long(update, msg)

    except Exception as e:
        await update.message.reply_text(f"💥 Erro no /relatorio: {type(e).__name__}: {e}")


async def cmd_categorias(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_row = _get_user_row(update.effective_user.id)
        if not user_row or not user_row.get("is_active"):
            await update.message.reply_text("Usuário não autorizado.")
            return
        account_id = user_row.get("account_id") or get_default_account_id()

        cats_rows = get_or_none(
            sb.table("categories").select("name").eq("account_id", account_id).order("name", desc=False).execute()
        ) or []

        if not cats_rows:
            await update.message.reply_text("📁 Ainda não tem categorias cadastradas.")
            return

        names = [r["name"] for r in cats_rows if r.get("name")]
        msg = "📁 Categorias:\n" + "\n".join([f"• {n}" for n in names])
        await _send_long(update, msg)
    except Exception as e:
        await update.message.reply_text(f"💥 Erro no /categorias: {type(e).__name__}: {e}")


async def cmd_extrato_obra(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = " ".join(context.args).strip()
    if not raw:
        await update.message.reply_text("Uso: /extrato_obra <nome> [período]. Ex: /extrato_obra João fevereiro")
        return

    period_markers = [
        "este mes", "essa semana", "semana passada", "mes passado",
        "ultimos", "últimos", "hoje", "ontem", "ultima quinzena", "última quinzena",
        "janeiro", "fevereiro", "marco", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
    ]

    low = _norm(raw)
    split_idx = None
    for mk in period_markers:
        pos = low.find(mk)
        if pos != -1:
            split_idx = pos
            break

    if split_idx is None:
        name = raw
        period_txt = ""
    else:
        name = raw[:split_idx].strip()
        period_txt = raw[split_idx:].strip()

    if not name:
        await update.message.reply_text("Me diz o nome da obra. Ex: /extrato_obra João fevereiro")
        return

    cc = f"OBRA_{_slugify_name(name)}"
    await run_cc_extrato(update, cc, period_txt)


async def cmd_ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📌 **BORIS — Ajuda rápida**\n"
        "——————————————\n"
        "✅ **Lançar despesas/receitas (texto ou áudio)**\n"
        "• \"paguei 150 de cabo elétrico na obra do Rodrigo\"\n"
        "• \"recebi mil do container do Thiago\"\n"
        "• \"paguei mil pra transportar o container da Ellen de munk\"\n\n"
        "✅ **Centro de custo (CC) entendido pelo texto**\n"
        "• Obra: \"obra do Rodrigo\" → `OBRA_RODRIGO`\n"
        "• Container: \"container do Thiago\" → `CONTAINER_THIAGO`\n"
        "• Bloco/Setor: \"bloco A\" → `BLOCO_A`\n"
        "• Sede: \"sede\"/\"adm\" → `SEDE`\n\n"
        "✅ **Consultas**\n"
        "• \"saldo do container do Thiago\" → pega todo o período\n"
        "• \"saldo do container do Thiago em fevereiro\" → pega fevereiro\n"
        "• \"extrato do container do Thiago\" → lista + resumo\n"
        "• \"extrato completo do container do Thiago\" → até 2000 itens\n"
        "• \"lucro do container do Thiago\"\n"
        "• \"margem da obra do Rodrigo\"\n"
        "• \"dre do container da Ellen\"\n"
        "• \"ranking de gastos\"\n"
        "• \"ranking de gastos em fevereiro\"\n\n"
        "✅ **Comandos**\n"
        "• /start\n"
        "• /autorizar <id>\n"
        "• /obra <nome>\n"
        "• /despesa <texto>\n"
        "• /receita <texto>\n"
        "• /saldo [período]\n"
        "• /relatorio\n"
        "• /resumo\n"
        "• /desfazer\n"
        "• /categorias\n"
        "• /extrato_obra <nome> [período]\n"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_user_text(update, context, update.message.text or "")


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
            text_out = ""
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
        await process_user_text(update, context, text_out)

    except Exception as e:
        msg = f"💥 Erro no handle_audio: {type(e).__name__}: {e}"
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            msg += "\n\nDica: manda de novo um áudio mais curto (até ~10s) ou manda em texto."
        await update.message.reply_text(msg)

# =====================================================================================
#                               TELEGRAM APP
# =====================================================================================

tg_app: Application = ApplicationBuilder().token(TOKEN).build()

tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("autorizar", cmd_autorizar))
tg_app.add_handler(CommandHandler("ajuda", cmd_ajuda))
tg_app.add_handler(CommandHandler("obra", cmd_obra))
tg_app.add_handler(CommandHandler("despesa", cmd_despesa))
tg_app.add_handler(CommandHandler("receita", cmd_receita))
tg_app.add_handler(CommandHandler("saldo", cmd_saldo))
tg_app.add_handler(CommandHandler("relatorio", cmd_relatorio))
tg_app.add_handler(CommandHandler("resumo", cmd_resumo))
tg_app.add_handler(CommandHandler("desfazer", cmd_desfazer))
tg_app.add_handler(CommandHandler("categorias", cmd_categorias))
tg_app.add_handler(CommandHandler("extrato_obra", cmd_extrato_obra))

tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, plain_text))
tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))


@app.on_event("startup")
async def on_startup():
    global _KEEPALIVE_TASK
    await tg_app.initialize()
    await tg_app.start()
    if _KEEPALIVE_TASK is None:
        _KEEPALIVE_TASK = asyncio.create_task(_daily_keepalive_loop())


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await tg_app.stop()
        await tg_app.shutdown()
    finally:
        pass

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


@app.get("/ping")
async def ping():
    ok = await _supabase_keepalive_once()
    return {"ok": True, "supabase": "ok" if ok else "fail"}


@app.get("/")
def alive():
    return {"boris": "ok"}
