import streamlit as st
from PIL import Image
import pytesseract
import re
import io
import os
import sqlite3
from datetime import datetime
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter
import string
import numpy as np
import requests
from requests.exceptions import RequestException, Timeout



# ---------- TRACK B: API CONFIG & SAFE REQUEST WRAPPER ----------

API_TIMEOUT = 8  # seconds

# Use Streamlit secrets for real deployment.
# In .streamlit/secrets.toml, you can keep:
#
# [api]
# FOREX_URL = "https://api.exchangerate.host/latest"
# FOREX_KEY = ""
# MARKET_URL = "https://your-market-api.example.com/quote"
# MARKET_KEY = "your_market_key"
# NEWS_URL = "https://your-news-api.example.com/search"
# NEWS_KEY = "your_news_key"

api_section = st.secrets.get("api", {}) if hasattr(st, "secrets") else {}

API_CONFIG = {
    "FOREX_URL": api_section.get("FOREX_URL", ""),
    "FOREX_KEY": api_section.get("FOREX_KEY", ""),
    "MARKET_URL": api_section.get("MARKET_URL", ""),
    "MARKET_KEY": api_section.get("MARKET_KEY", ""),
    "NEWS_URL": api_section.get("NEWS_URL", ""),
    "NEWS_KEY": api_section.get("NEWS_KEY", ""),
}


def safe_get_json(url, params=None, headers=None):
    """
    Generic GET helper with strong error handling.
    Returns: (data_dict_or_list_or_None, error_message_or_None)
    """
    if not url:
        return None, "API endpoint not configured. Please set it in Streamlit secrets under [api]."

    if params is None:
        params = {}
    if headers is None:
        headers = {}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=API_TIMEOUT)
    except Timeout:
        return None, "Request timed out while calling the API. Please try again later."
    except RequestException as e:
        return None, f"Network error while calling the API: {e}"

    if resp.status_code != 200:
        body_snippet = resp.text[:200]
        return None, f"API returned HTTP {resp.status_code}. Response: {body_snippet}"

    try:
        data = resp.json()
    except ValueError:
        return None, "API returned invalid JSON. Please check provider or parameters."
    return data, None



# ---------- TRACK B: EXAMPLE FINANCIAL API WRAPPERS ----------

def fetch_forex_inr_snapshot():
    """
    Get INR exchange rates for a few major currencies.
    Example provider: https://api.exchangerate.host/latest?base=INR
    Configure FOREX_URL accordingly.
    """
    base_url = API_CONFIG["FOREX_URL"]
    params = {}

    data, err = safe_get_json(base_url, params=params)
    if err:
        return None, err

    # Adapt this to your actual provider structure
    rates = data.get("rates") or data.get("data") or {}
    if not isinstance(rates, dict):
        return None, "Unexpected forex API format."

    wanted = ["USD", "EUR", "GBP", "JPY"]
    snapshot = {cur: rates.get(cur) for cur in wanted if cur in rates}
    if not snapshot:
        return None, "Forex API did not return expected INR-based rates."
    return snapshot, None


def fetch_market_index_snapshot(symbol="NIFTY50"):
    """
    Fetch an index/stock snapshot.
    MARKET_URL should be configured to your provider.
    For demo, we just pass `symbol` as a query param.
    """
    base_url = API_CONFIG["MARKET_URL"]
    params = {"symbol": symbol}
    headers = {}

    # Example: if provider requires a key in header
    if API_CONFIG["MARKET_KEY"]:
        headers["Authorization"] = f"Bearer {API_CONFIG['MARKET_KEY']}"

    data, err = safe_get_json(base_url, params=params, headers=headers)
    if err:
        return None, err

    if not isinstance(data, dict):
        return None, "Unexpected market API format."

    # Expect something like: {"symbol": "NIFTY50", "price": 22000, "change_pct": -0.5}
    return data, None


def fetch_mutual_fund_nav(fund_code: str):
    """
    Mutual fund NAV lookup (demo).
    You can point MARKET_URL to an MF NAV API, passing `fund_code`.
    """
    base_url = API_CONFIG["MARKET_URL"]
    params = {"fund": fund_code}

    data, err = safe_get_json(base_url, params=params)
    if err:
        return None, err

    if not isinstance(data, dict):
        return None, "Unexpected mutual fund API format."

    # Expect: {"fund": "...", "nav": 123.45, "date": "2025-01-01"}
    return data, None


def fetch_financial_news(query: str = "personal finance India"):
    """
    Financial news/articles search.
    Configure NEWS_URL & NEWS_KEY in secrets.
    """
    base_url = API_CONFIG["NEWS_URL"]
    params = {"q": query}
    headers = {}

    if API_CONFIG["NEWS_KEY"]:
        # Depends on provider, this is one common pattern
        headers["X-API-KEY"] = API_CONFIG["NEWS_KEY"]

    data, err = safe_get_json(base_url, params=params, headers=headers)
    if err:
        return None, err

    # Expect: {"articles": [ {...}, {...} ]}
    articles = data.get("articles")
    if not isinstance(articles, list):
        return None, "Unexpected news API format."

    return articles[:5], None  # top 5






# ---------- INDIAN CURRENCY FORMATTER ----------

def format_inr(amount: float, decimals: int = 0) -> str:
    """
    Format numbers like 1,23,456.78 with ₹ sign.
    decimals = 0 or 2 usually.
    """
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return "₹0"

    negative = amt < 0
    amt = abs(amt)

    rupees = int(amt)
    paise = int(round((amt - rupees) * (10 ** decimals))) if decimals > 0 else 0

    s = str(rupees)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        s = ",".join(reversed(parts)) + "," + last3

    if decimals > 0:
        s = f"{s}.{paise:0{decimals}d}"

    prefix = "-₹" if negative else "₹"
    return prefix + s





# ---------- ML EXPENSE CATEGORIZATION MODEL ----------

def load_ml_model():
    try:
        df = pd.read_csv("expense_training_data.csv")

        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(df["description"])
        y = df["category"]

        model = LogisticRegression()
        model.fit(X, y)

        return model, vectorizer
    except Exception as e:
        st.warning(f"ML model could not load: {e}")
        return None, None


ml_model, ml_vectorizer = load_ml_model()


def ml_predict_category(text: str):
    if not ml_model:
        return None

    X = ml_vectorizer.transform([text])
    prediction = ml_model.predict(X)[0]

    # Optional: check model confidence
    proba = max(ml_model.predict_proba(X)[0])
    if proba < 0.50:  # low confidence → fallback
        return None

    return prediction



# ---------- FINANCIAL GURU PROFILES ----------

GURU_PROFILES = {
    "Warren Buffett": {
        "summary": "Focus on long-term investing, living below your means, and avoiding unnecessary debt.",
        "core_rules": [
            "Spend less than you earn and avoid lifestyle inflation.",
            "Avoid high-interest debt and unnecessary EMIs.",
            "Prefer long-term investments in quality assets over frequent trading.",
        ],
        "focus_categories": ["Shopping", "Entertainment", "Others"],
    },
    "Robert Kiyosaki": {
        "summary": "Separate assets from liabilities and focus on building cash-flowing assets.",
        "core_rules": [
            "Reduce spending on liabilities that don’t put money back in your pocket.",
            "Use savings to acquire assets (investments, skills, businesses), not just consumption.",
            "Track your cash flow carefully and avoid ‘invisible’ small leaks.",
        ],
        "focus_categories": ["Shopping", "Transport", "Bills"],
    },
    "Ramit Sethi": {
        "summary": "Build a conscious spending plan: aggressively cut costs on things you don’t care about and spend more on what you love.",
        "core_rules": [
            "Automate savings and investments every month.",
            "Define 3–5 ‘money dials’ you love (travel, food, gadgets) and consciously fund them.",
            "Cut subscriptions and random spending you don’t truly value.",
        ],
        "focus_categories": ["Food", "Entertainment", "Shopping", "Bills"],
    },
    "Indian Tax-Saving Focus": {
        "summary": "Use Indian instruments like PPF, ELSS, and SIPs to reduce tax and build long-term wealth.",
        "core_rules": [
            "Target investing 10–20% of income into SIPs / long-term equity funds.",
            "Use Section 80C options (PPF, ELSS, EPF, life insurance) effectively before year-end.",
            "Avoid short-term trading; prefer systematic, long-term plans.",
        ],
        "focus_categories": ["Bills", "Others"],
    },
}


# ---------- DEFAULT "BEST BOOK" STYLE (used when no good upload) ----------

DEFAULT_BOOK_PROFILE = {
    "title": "The Psychology of Money (default book)",
    "summary": (
        "Focus on long-term thinking, controlling emotions around money, and building "
        "steady habits like consistent saving and simple investing."
    ),
    "core_rules": [
        "Avoid lifestyle inflation; let your savings rate grow as your income grows.",
        "Prioritize long-term, simple investments over frequent trading or speculation.",
        "Build an emergency fund so you never have to make money decisions in panic.",
    ],
    "focus_categories": ["Entertainment", "Shopping", "Food", "Others"],
}




# ---------- DEFAULT FINANCIAL BOOK PROFILES (used if user upload is missing/useless) ----------

DEFAULT_BOOK_PROFILES = {
    "The Intelligent Investor": {
        "summary": "Classic investing book focusing on long-term, value-based investing, risk management, and not speculating.",
        "key_ideas": [
            "Treat stocks as ownership in real businesses, not lottery tickets.",
            "Focus on margin of safety: don't overpay for risky or overhyped assets.",
            "Be a disciplined, long-term investor instead of reacting emotionally to market moves.",
        ],
        "best_for_themes": ["Investing & Assets", "Mindset & Habits"],
    },
    "Rich Dad Poor Dad": {
        "summary": "Popular personal finance book focused on assets vs liabilities, cash flow, and financial education.",
        "key_ideas": [
            "Buy assets that put money into your pocket, reduce liabilities that take money out.",
            "Use your job income to build or buy income-generating assets over time.",
            "Invest in your financial education so you understand money, not just work for it.",
        ],
        "best_for_themes": ["Investing & Assets", "Debt & Loans", "Income & Career Growth"],
    },
    "I Will Teach You To Be Rich": {
        "summary": "Modern Indian-friendly personal finance book focused on conscious spending, automation, and simple investing.",
        "key_ideas": [
            "Create a conscious spending plan: aggressively cut what you don't care about and spend more on what you love.",
            "Automate savings and investments so they happen every month without willpower.",
            "Use simple investing (SIPs / index funds) instead of trying to outsmart the market.",
        ],
        "best_for_themes": ["Budgeting & Expense Control", "Investing & Assets", "Mindset & Habits"],
    },
}

def choose_default_book(selected_gurus, guru_analysis):
    """
    Pick a default high-quality book profile when:
      - user didn't upload content OR
      - uploaded content is low quality
    Preference:
      - match themes from guru_analysis, else
      - match selected guru, else
      - fallback to 'I Will Teach You To Be Rich'
    """
    # 1) Try matching by themes from uploaded text (if any)
    if guru_analysis and guru_analysis.get("themes"):
        themes = set(guru_analysis["themes"])
        best_name = None
        best_overlap = -1
        for name, prof in DEFAULT_BOOK_PROFILES.items():
            overlap = len(themes.intersection(set(prof["best_for_themes"])))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name
        if best_name:
            return best_name, DEFAULT_BOOK_PROFILES[best_name]

    # 2) Try matching selected guru
    if selected_gurus:
        gurus_lower = [g.lower() for g in selected_gurus]
        if any("ramit" in g for g in gurus_lower):
            return "I Will Teach You To Be Rich", DEFAULT_BOOK_PROFILES["I Will Teach You To Be Rich"]
        if any("kiyosaki" in g for g in gurus_lower):
            return "Rich Dad Poor Dad", DEFAULT_BOOK_PROFILES["Rich Dad Poor Dad"]
        if any("buffett" in g for g in gurus_lower):
            return "The Intelligent Investor", DEFAULT_BOOK_PROFILES["The Intelligent Investor"]

    # 3) Final fallback
    return "I Will Teach You To Be Rich", DEFAULT_BOOK_PROFILES["I Will Teach You To Be Rich"]





# ---------- FINANCIAL CONTENT THEMES (for uploaded books/articles) ----------

FINANCIAL_THEMES = {
    "Budgeting & Expense Control": [
        "budget", "budgeting", "expense", "spending", "save", "saving", "cut costs",
        "frugal", "frugality", "overspending"
    ],
    "Investing & Assets": [
        "invest", "investment", "investing", "stocks", "equity", "mutual fund",
        "sip", "asset", "assets", "portfolio", "return", "compound"
    ],
    "Debt & Loans": [
        "debt", "loan", "loans", "interest", "emi", "credit card", "repay",
        "repayment", "liability"
    ],
    "Emergency Fund & Safety": [
        "emergency fund", "rainy day", "safety", "buffer", "contingency"
    ],
    "Income & Career Growth": [
        "salary", "income", "earn", "earning", "side hustle", "career",
        "skills", "promotion", "freelance"
    ],
    "Mindset & Habits": [
        "mindset", "habit", "discipline", "behavior", "psychology", "mind",
        "routine", "consistency"
    ],
}

def analyze_guru_text(text: str):
    """
    Analyze uploaded financial books/articles text to extract:
      - word count
      - top keywords
      - detected themes (budget, investing, debt, etc.)
      - is_useful: whether this looks like real financial content
    """
    if not text or not text.strip():
        return None

    # basic cleaning
    lower = text.lower()
    for ch in string.punctuation:
        lower = lower.replace(ch, " ")

    words = [w for w in lower.split() if len(w) > 3]
    if not words:
        return None

    word_count = len(words)

    # simple keyword frequency
    counter = Counter(words)
    stopwords = {
        "this", "that", "with", "from", "have", "will", "your", "about", "which",
        "when", "where", "there", "their", "them", "then", "they", "what", "into",
        "also", "because", "would", "could", "should", "over", "under", "very",
        "more", "less", "than", "only", "some", "many", "these", "those", "such",
        "most", "like", "just"
    }
    for sw in stopwords:
        counter.pop(sw, None)

    top_keywords = counter.most_common(15)

    # detect themes based on FINANCIAL_THEMES keywords
    detected_themes = []
    for theme, kws in FINANCIAL_THEMES.items():
        for kw in kws:
            if kw in lower:
                detected_themes.append(theme)
                break
    detected_themes = sorted(set(detected_themes))

    approx_pages = max(word_count // 300, 1)  # ~300 words per page

    # 🔍 decide if this is a "useful" book/article:
    #   - enough length
    #   - at least one financial theme detected
    is_useful = (word_count >= 800) and bool(detected_themes)

    return {
        "word_count": word_count,
        "approx_pages": approx_pages,
        "top_keywords": top_keywords,
        "themes": detected_themes,
        "is_useful": is_useful,
    }




def analyze_multiple_guru_docs(files):
    """
    Process multiple uploaded financial books/articles (PDF/TXT).

    Returns:
      - combined_text: str  (all text merged, trimmed to ~20k chars)
      - combined_analysis: dict or None, with keys:
            word_count, approx_pages, top_keywords, themes, is_useful,
            sources: [
                {
                    "name": filename,
                    "word_count": int,
                    "approx_pages": int,
                    "themes": [...],
                    "is_useful": bool,
                },
                ...
            ],
            primary_source: best useful file name or None
    """
    if not files:
        return "", None

    per_sources = []
    texts = []

    for f in files:
        name = f.name
        fname_lower = name.lower()
        text = ""

        try:
            # TXT file
            if fname_lower.endswith(".txt"):
                f.seek(0)
                text = f.read().decode("utf-8", errors="ignore")

            # PDF file
            elif fname_lower.endswith(".pdf"):
                try:
                    reader = PdfReader(io.BytesIO(f.getvalue()))
                    pages_text = []
                    for page in reader.pages:
                        try:
                            page_text = page.extract_text() or ""
                        except Exception:
                            page_text = ""
                        pages_text.append(page_text)
                    text = "\n".join(pages_text)
                except Exception:
                    text = ""
        except Exception as e:
            st.warning(f"Could not read {name}: {e}")
            text = ""

        if not text or not text.strip():
            # still create an entry so we know it existed
            per_sources.append({
                "name": name,
                "word_count": 0,
                "approx_pages": 0,
                "themes": [],
                "is_useful": False,
            })
            continue

        # Run single-text analysis for this file
        single_analysis = analyze_guru_text(text) or {
            "word_count": 0,
            "approx_pages": 0,
            "top_keywords": [],
            "themes": [],
            "is_useful": False,
        }

        per_sources.append({
            "name": name,
            "word_count": single_analysis.get("word_count", 0),
            "approx_pages": single_analysis.get("approx_pages", 0),
            "themes": single_analysis.get("themes", []),
            "is_useful": single_analysis.get("is_useful", False),
        })

        texts.append(text)

    if not texts:
        # nothing readable
        return "", None

    combined_text = "\n".join(texts)
    # keep size reasonable
    if len(combined_text) > 20000:
        combined_text = combined_text[:20000]

    combined_analysis = analyze_guru_text(combined_text) or {
        "word_count": 0,
        "approx_pages": 0,
        "top_keywords": [],
        "themes": [],
        "is_useful": False,
    }

    # Merge/upgrade with per-source info
    combined_analysis["sources"] = per_sources

    # Sum pages & word counts from all sources if bigger
    total_wc = sum(s["word_count"] for s in per_sources)
    total_pages = sum(s["approx_pages"] for s in per_sources)
    if total_wc > combined_analysis.get("word_count", 0):
        combined_analysis["word_count"] = total_wc
    if total_pages > combined_analysis.get("approx_pages", 0):
        combined_analysis["approx_pages"] = total_pages

    # Merge themes (union)
    all_themes = set(combined_analysis.get("themes", []))
    for s in per_sources:
        all_themes.update(s.get("themes", []))
    combined_analysis["themes"] = sorted(all_themes)

    # If ANY source is useful -> mark overall as useful
    if any(s.get("is_useful") for s in per_sources):
        combined_analysis["is_useful"] = True

    # Primary source = biggest useful document
    useful_sources = [s for s in per_sources if s.get("is_useful")]
    if useful_sources:
        primary = max(useful_sources, key=lambda s: s["word_count"])
        combined_analysis["primary_source"] = primary["name"]
    else:
        combined_analysis["primary_source"] = None

    return combined_text, combined_analysis




# ---------- CUSTOM CATEGORY LEARNING (user-trained) ----------

CUSTOM_CATEGORY_MAP = {}  # pattern (str) -> category (str)

def load_custom_category_map():
    """
    Load previously learned custom category patterns from CSV.
    File format: pattern,category
    """
    global CUSTOM_CATEGORY_MAP
    path = "custom_category_learning.csv"
    if not os.path.exists(path):
        CUSTOM_CATEGORY_MAP = {}
        return

    try:
        df_map = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read custom_category_learning.csv: {e}")
        CUSTOM_CATEGORY_MAP = {}
        return

    mapping = {}
    for _, row in df_map.iterrows():
        patt = str(row.get("pattern", "")).strip().lower()
        cat = str(row.get("category", "")).strip()
        if patt and cat:
            mapping[patt] = cat

    CUSTOM_CATEGORY_MAP = mapping


def save_custom_category_map():
    """
    Save current CUSTOM_CATEGORY_MAP to CSV so it persists across sessions.
    """
    path = "custom_category_learning.csv"
    if not CUSTOM_CATEGORY_MAP:
        # if empty, you can choose to remove file or just leave as is
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return

    rows = [{"pattern": k, "category": v} for k, v in CUSTOM_CATEGORY_MAP.items()]
    df_map = pd.DataFrame(rows)
    df_map.drop_duplicates(subset=["pattern"], keep="last", inplace=True)

    try:
        df_map.to_csv(path, index=False)
    except Exception as e:
        st.warning(f"Could not save custom_category_learning.csv: {e}")





# ---------- CATEGORY KEYWORDS (rule-based) ----------
CATEGORY_KEYWORDS = {
    "Food": [
        "zomato", "swiggy", "restaurant", "hotel", "pizza", "instamart",
        "meal", "burger", "kfc", "mcd", "cafe", "breakfast", "lunch", "dinner"
    ],
    "Groceries": [
        "grocery", "supermarket", "dmart", "mart", "bazaar", "bigbasket",
        "ratnadeep", "more supermarket", "reliance fresh"
    ],
    "Transport": [
        "uber", "ola", "taxi", "bus", "train", "fuel", "petrol", "diesel",
        "auto", "metro", "cab", "ola cab", "rapido", "parking"
    ],
    "Shopping": [
        "amazon", "flipkart", "ajio", "myntra", "mall", "store", "shop",
        "fashion", "cloth", "purchase", "reliance trends", "lifestyle",
        "croma", "vijay sales", "electronics"
    ],
    "Entertainment": [
        "movie", "netflix", "spotify", "pvr", "inox", "ticket", "show",
        "game", "hotstar", "bookmyshow", "zee5", "prime video"
    ],
    "Health": [
        "pharmacy", "medical", "medicine", "hospital", "clinic", "doctor",
        "chemist", "apollo", "medplus", "lab test", "diagnostic"
    ],
    "Bills": [
        "electricity", "bill", "wifi", "broadband", "recharge", "dth",
        "prepaid", "postpaid", "jio", "airtel", "vi", "bsnl", "gas bill",
        "water bill"
    ],
    "Rent": [
        "rent", "pg", "hostel", "room rent", "flat rent", "house rent",
        "owner rent", "landlord"
    ],
    "Education": [
        "tuition", "college", "fee", "exam", "university", "coaching",
        "course", "udemy", "coursera", "byjus", "unacademy", "school"
    ],
    "Loans & EMIs": [
        "emi", "loan", "personal loan", "home loan", "car loan",
        "credit card due", "nbfc", "hdfc bank loan", "icici loan", "sbi loan"
    ],
    "Investments": [
        "sip", "mutual fund", "ppf", "elss", "nps", "lic", "fd", "rd",
        "recurring deposit", "fixed deposit", "demat", "zerodha", "groww",
        "et money", "stock", "shares", "index fund"
    ],
    "Others": []
}


# ---------- SMALL ML MODEL FOR CATEGORY PREDICTION ----------

ML_TRAINING_DATA = [
    ("Food", "Food"),
    ("zomato food order", "Food"),
    ("swiggy instamart groceries", "Food"),
    ("dominos pizza", "Food"),
    ("kfc chicken", "Food"),
    ("mcdonalds burger", "Food"),
    ("bigbasket grocery", "Groceries"),
    ("dmart supermarket", "Groceries"),

    # ("Treval", "Transport")
    ("uber ride", "Transport"),
    ("ola cab", "Transport"),
    ("metro ticket", "Transport"),
    ("bus pass", "Transport"),
    ("petrol pump hpcl", "Transport"),

    ("amazon shopping", "Shopping"),
    ("flipkart order", "Shopping"),
    ("myntra clothes", "Shopping"),
    ("ajio fashion", "Shopping"),
    ("reliance trends purchase", "Shopping"),

    ("movie ticket pvr", "Entertainment"),
    ("bookmyshow movie", "Entertainment"),
    ("netflix subscription", "Entertainment"),
    ("hotstar premium", "Entertainment"),
    ("spotify monthly", "Entertainment"),

    ("apollo pharmacy", "Health"),
    ("medplus medical", "Health"),
    ("doctor consultation", "Health"),
    ("hospital bill", "Health"),

    ("jio mobile recharge", "Bills"),
    ("airtel broadband bill", "Bills"),
    ("vi prepaid recharge", "Bills"),
    ("electricity bill mescom", "Bills"),
    ("wifi bill", "Bills"),
]

# Train once when app starts
def train_ml_model():
    texts = [t for (t, c) in ML_TRAINING_DATA]
    labels = [c for (t, c) in ML_TRAINING_DATA]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return model, vectorizer

ML_MODEL, ML_VECTORIZER = train_ml_model()

def ml_predict_category(text: str):
    """Predict category using ML model. Returns None if not confident."""
    if not text.strip():
        return None
    X = ML_VECTORIZER.transform([text])
    proba = ML_MODEL.predict_proba(X)[0]
    pred = ML_MODEL.classes_[proba.argmax()]
    confidence = proba.max()
    # lower threshold so it actually predicts
    if confidence < 0.35:
        return None
    return pred



# Load any previously learned custom categories on app start
load_custom_category_map()





# ---------- FINAL CATEGORY FUNCTION (ML + rules) ----------
def detect_category(text_line: str) -> str:
    t = text_line.lower()

    # 0) CUSTOM LEARNED PATTERNS (from user corrections)
    # If any learned pattern appears in this line, use that category immediately.
    for patt, cat in CUSTOM_CATEGORY_MAP.items():
        if patt and patt in t:
            return cat

    # 1) RULE-BASED (keywords), like your original version
    best_category = "Others"
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for k in keywords if k in t)
        if score > best_score:
            best_score = score
            best_category = category

    if best_score > 0:
        return best_category

    # 2) ML fallback if rules didn’t find anything
    ml_cat = ml_predict_category(t)
    if ml_cat:
        return ml_cat

    # 3) Default
    return "Others"



# ---------- CATEGORY ADVICE DATABASE ----------
CATEGORY_ADVICE = {
    "Food": [
        "🍽️ Keep food spending under 15% of your total income.",
        "🍱 Plan weekly meals to avoid overspending on takeout.",
        "☕ Make coffee at home — small savings add up!"
    ],
    "Transport": [
        "🚗 Track your fuel usage and plan trips efficiently.",
        "🚌 Try carpooling or public transport to save fuel costs.",
        "🚴 Short distances? Consider walking or biking!"
    ],
    "Shopping": [
        "🛍️ Avoid impulse buys — wait 24 hours before purchasing.",
        "💳 Compare prices before checking out online.",
        "📦 Track your monthly shopping budget and set limits."
    ],
    "Entertainment": [
        "🎬 Limit subscriptions to only the services you use often.",
        "🎮 Budget for entertainment — 5–10% of income max.",
        "🎧 Free hobbies can be just as rewarding!"
    ],
    "Health": [
        "💊 Keep 10% of your income aside for medical expenses.",
        "🏥 Health is wealth — insurance saves in emergencies.",
        "🧘‍♂️ Invest in preventive care, not just medicine."
    ],
    "Groceries": [
        "🛒 Make a weekly grocery list and stick to it.",
        "🥦 Buy in bulk for non-perishables to save money.",
        "🍎 Compare prices across stores for essentials."
    ],
    "Bills": [
        # "💡 Automate payments to avoid late fees.",
        "📶 Check subscriptions and cancel unused plans.",
        "📲 Review recurring payments every month."
    ],
    "Others": [
        "📘 Track unknown expenses manually to avoid leaks.",
        "💰 Every rupee counts — even small savings matter."
    ]
}

# ---------- ADVANCED OCR HELPERS (multi-format) ----------

# If you want, you can hardcode poppler path here (for Windows PDF support)
POPPLER_PATH = r"C:\Program Files\poppler\Library\bin"  # 👈 change this to your real poppler bin path, or set to None

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "tiff", "bmp", "pdf"]


def extract_text_from_image(image):
    """Run Tesseract OCR on a single image."""
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(file_bytes):
    """
    Convert PDF pages to images using pdf2image + poppler,
    then run OCR page by page and combine the text.
    """
    kwargs = {}
    if POPPLER_PATH and os.path.isdir(POPPLER_PATH):
        kwargs["poppler_path"] = POPPLER_PATH

    # Convert all pages to PIL Images
    pages = convert_from_bytes(file_bytes, dpi=300, **kwargs)

    all_text = []
    for i, page in enumerate(pages, start=1):
        page_text = pytesseract.image_to_string(page)
        all_text.append(f"\n\n===== PAGE {i} =====\n\n{page_text}")

    return "\n".join(all_text), pages  # return all text + list of page images


def advanced_ocr(uploaded_file):
    """
    Detect file type (image / pdf) and route to correct OCR pipeline.
    Returns (text, preview_image)
    """
    file_bytes = uploaded_file.getvalue()
    filename = (uploaded_file.name or "").lower()

    # Decide by extension + mime
    is_pdf = filename.endswith(".pdf") or uploaded_file.type == "application/pdf"

    if is_pdf:
        text, pages = extract_text_from_pdf(file_bytes)
        preview_image = pages[0] if pages else None
        return text, preview_image
    else:
        # Generic image (png, jpg, jpeg, webp, tiff, bmp, etc.)
        image = Image.open(io.BytesIO(file_bytes))
        text = extract_text_from_image(image)
        return text, image


# ---------- Helper: Find category ----------
# def detect_category(text):
#     text = text.lower().strip()

#     # 1️⃣ Try ML prediction
#     ml_cat = ml_predict_category(text)
#     if ml_cat:
#         return ml_cat

#     # 2️⃣ fallback to rule-based
#     CATEGORY_KEYWORDS = {
#         "Food": ["zomato", "swiggy", "restaurant", "pizza", "instamart"],
#         "Transport": ["uber", "ola", "fuel", "petrol", "diesel", "cab"],
#         "Bills": ["recharge", "airtel", "vi", "jio", "electricity", "bill"],
#         "Shopping": ["amazon", "flipkart", "myntra", "ajio"],
#         "Entertainment": ["movie", "netflix", "spotify", "hotstar"],
#     }

#     for cat, keywords in CATEGORY_KEYWORDS.items():
#         if any(k in text for k in keywords):
#             return cat

#     return "Other"


def looks_like_fee_receipt(text: str) -> bool:
    """
    Detects big tuition/college style receipts where we usually
    only care about ONE total amount.
    """
    t = text.lower()
    keywords = ["tuition", "college", "university", "exam fee", "admission", "receipt no"]
    return any(k in t for k in keywords)



# ---------- Helper: Parse each line ----------
def parse_expenses(text):
    expenses = []
    lines = text.splitlines()
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        # find amount
        matches = re.findall(r'([0-9]+(?:\.[0-9]{1,2})?)', clean)
        if not matches:
            continue
        amount = float(matches[-1])
        category = detect_category(clean)
        # remove digits to keep only item name
        name = re.sub(r'[^a-zA-Z ]', '', clean).strip().title()
        expenses.append({
            "Item": name if name else "Unknown",
            "Amount (₹)": amount,
            "Category": category
        })
    return expenses


def looks_like_bank_statement(text: str) -> bool:
    """
    Simple heuristic: if many lines start with a date, treat it as bank statement.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    date_pattern = re.compile(r'^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')  # e.g. 01/12/2025 or 01-12-25
    date_lines = sum(1 for l in lines if date_pattern.match(l))
    return date_lines >= 3  # 3+ lines with date => likely statement


def parse_bank_statement(text: str):
    """
    Parse bank statement-style text into structured transactions:
    Date, Description, Amount, Type (Debit/Credit), Category
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    date_pattern = re.compile(r'^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')  # basic dd/mm/yyyy or dd-mm-yyyy

    transactions = []

    for line in lines:
        m = date_pattern.match(line)
        if not m:
            continue  # skip lines that don't start with a date

        date_str = m.group(1)

        # Get all numeric values in the line (amount is usually last)
        nums = re.findall(r'([0-9]+(?:\.[0-9]{1,2})?)', line)
        if not nums:
            continue

        amount = float(nums[-1])

        # Guess type: Debit or Credit
        lower = line.lower()
        if " cr" in lower or "credit" in lower:
            tx_type = "Credit"
        elif " dr" in lower or "debit" in lower:
            tx_type = "Debit"
        else:
            # default: treat as debit (money going out)
            tx_type = "Debit"

        # Description = everything between date and amount
        desc_part = line[m.end():]
        # remove the last numeric piece from description
        desc_part = re.sub(r'([0-9]+(?:\.[0-9]{1,2})?)\s*$', "", desc_part)

        # Clean for category + name
        desc_alpha = re.sub(r'[^a-zA-Z ]', ' ', desc_part).strip()
        description = desc_alpha.title() if desc_alpha else "Transaction"

        category = detect_category(desc_part)  # reuse your keyword logic

        transactions.append({
            "Date": date_str,
            "Description": description,
            "Amount (₹)": amount,
            "Type": tx_type,
            "Category": category,
        })

    return transactions




def looks_like_upi_block(text: str) -> bool:
    """
    Detects if the given text looks like UPI / Indian payment app messages.
    Checks for common markers like 'upi', '@oksbi', 'gpay', 'phonepe', 'paytm', etc.
    """
    t = text.lower()
    keywords = [
        "upi", "vpa", "@oksbi", "@okaxis", "@okhdfcbank", "@okicici",
        "google pay", "gpay", "phonepe", "paytm", "bhim upi", "upi ref", "upi txn"
    ]
    return any(k in t for k in keywords)


def parse_upi_transactions(text: str):
    """
    Parse UPI-style lines into structured expenses:
    Item, Amount (₹), Category, Type, Channel
    """
    expenses = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        low = line.lower()
        if not looks_like_upi_block(low):
            continue

        # Find amount (₹, Rs, INR formats)
        matches = re.findall(r'([0-9]+(?:\.[0-9]{1,2})?)', line.replace(",", ""))
        if not matches:
            continue
        amount = float(matches[-1])

        # Try to guess channel
        channel = None
        if "gpay" in low or "google pay" in low:
            channel = "Google Pay"
        elif "phonepe" in low:
            channel = "PhonePe"
        elif "paytm" in low:
            channel = "Paytm"
        elif "bhim" in low:
            channel = "BHIM UPI"
        else:
            channel = "UPI"

        # Extract a clean item/payee name
        # Remove common noise words and numbers/symbols
        noise_words = [
            "upi", "payment", "paid to", "credited to", "debited from",
            "via", "through", "txn", "transaction", "ref", "refno", "no", "rs", "inr"
        ]
        cleaned = line
        for w in noise_words:
            cleaned = re.sub(w, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'[^a-zA-Z ]', ' ', cleaned).strip()
        item_name = "UPI Payment"
        if cleaned:
            # keep only first few words as label
            tokens = [t for t in cleaned.split() if len(t) > 2]
            if tokens:
                item_name = " ".join(tokens[:4]).title()

        category = detect_category(line)

        expenses.append({
            "Item": item_name,
            "Amount (₹)": amount,
            "Category": category,
            "Type": "UPI Debit",
            "Channel": channel,
        })

    return expenses






def parse_receipt_total(text: str):
    """
    For receipt-style documents:
    - Scan the whole text
    - Find all numbers
    - Keep only 'reasonable' amounts
    - Take the largest one as the main receipt total
    - Return ONE expense row
    """
    matches = re.findall(r'([0-9]{2,}(?:\.[0-9]{1,2})?)', text)
    if not matches:
        return []

    amounts = [float(m) for m in matches]

    # Filter out very small / crazy huge values
    filtered = [a for a in amounts if 10 <= a <= 1_000_000]
    if not filtered:
        return []

    total = max(filtered)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    item_name = "Receipt Total"
    line_for_cat = None
    target_str = str(int(total))

    for ln in lines:
        if target_str in ln.replace(",", ""):
            clean = re.sub(r'[^a-zA-Z ]', ' ', ln).strip().title()
            if clean:
                item_name = clean
            line_for_cat = ln
            break

    # Use the matching line (not full text) for category
    if line_for_cat is None:
        line_for_cat = text
    category = detect_category(line_for_cat)

    return [{
        "Item": item_name,
        "Amount (₹)": total,
        "Category": category
    }]



def looks_like_payment_message_block(text: str) -> bool:
    """
    Detects SMS-style payment alerts from banks / cards.
    """
    t = text.lower()
    keywords = [
        "debited from", "credited to", "spent on your", "txn", "transaction",
        "available balance", "a/c", "account", "card ending", "card no", "imps", "neft", "rtgs"
    ]
    return any(k in t for k in keywords)


def parse_payment_sms_messages(text: str):
    """
    Parse SMS-style payment messages into:
      Item (merchant), Amount (₹), Category, Type, Channel
    """
    expenses = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in lines:
        low = ln.lower()
        if not looks_like_payment_message_block(low):
            continue

        # Amount
        matches = re.findall(r'([0-9]+(?:\.[0-9]{1,2})?)', ln.replace(",", ""))
        if not matches:
            continue
        amount = float(matches[-1])

        # Type: Debit or Credit
        if "credited" in low or "credit" in low:
            tx_type = "Credit"
        else:
            tx_type = "Debit"

        # Channel guess
        if "credit card" in low or "card" in low:
            channel = "Card"
        elif "imps" in low:
            channel = "IMPS"
        elif "neft" in low:
            channel = "NEFT"
        elif "rtgs" in low:
            channel = "RTGS"
        else:
            channel = "Bank Txn"

        # Merchant / payee extraction (simple heuristics)
        merchant = "Payment Transaction"
        # Common pattern: "at AMAZON" or "at BigBazaar"
        m = re.search(r'at\s+([A-Za-z0-9 &]+)', ln, flags=re.IGNORECASE)
        if m:
            merchant = m.group(1).strip().title()
        else:
            # fallback: strip digits and keep some words
            cleaned = re.sub(r'[^a-zA-Z ]', ' ', ln).strip()
            tokens = [t for t in cleaned.split() if len(t) > 2]
            if tokens:
                merchant = " ".join(tokens[-3:]).title()

        category = detect_category(ln)

        expenses.append({
            "Item": merchant,
            "Amount (₹)": amount,
            "Category": category,
            "Type": tx_type,
            "Channel": channel,
        })

    return expenses





def looks_like_itemized_receipt(text: str) -> bool:
    """
    Detects shop-style receipts with multiple line items and a final total.
    """
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    amount_lines = 0
    total_lines = 0

    for ln in lines:
        if re.search(r'([0-9]+(?:\.[0-9]{1,2})?)', ln.replace(",", "")):
            amount_lines += 1
        if any(k in ln for k in ["total", "grand total", "amount payable", "net amount"]):
            total_lines += 1

    # Heuristic: multiple amount lines + at least one "total"
    return amount_lines >= 3 and total_lines >= 1


def parse_itemized_receipt(text: str):
    """
    Parse itemized receipts:
      - Each line with an amount is treated as an item,
      - But 'Total', 'GST', 'Tax', 'Amount Payable' lines are ignored
      - Avoid double counting overall totals.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items = []

    ignore_keywords = [
        "total", "grand total", "gst", "tax", "cgst", "sgst",
        "amount payable", "net amount", "round off"
    ]

    for ln in lines:
        low = ln.lower()
        # skip obvious summary lines
        if any(k in low for k in ignore_keywords):
            continue

        matches = re.findall(r'([0-9]+(?:\.[0-9]{1,2})?)', ln.replace(",", ""))
        if not matches:
            continue

        amount = float(matches[-1])

        # Build item name from alphabets only
        name = re.sub(r'[^a-zA-Z ]', ' ', ln).strip().title()
        if not name:
            name = "Receipt Item"

        category = detect_category(ln)

        items.append({
            "Item": name,
            "Amount (₹)": amount,
            "Category": category,
        })

    # If we found at least 2 item lines, use them.
    if len(items) >= 2:
        return items

    # Otherwise fallback to single total extraction
    return parse_receipt_total(text)



def build_predictive_financial_model(df: pd.DataFrame):
    """
    Predict:
     - Next-month total spending
     - Category-level drift (increase/decrease %)
     - Future saving ability
     - High-risk overspending categories

    Returns dictionary with prediction components.
    """
    if df is None or df.empty:
        return {}

    df2 = df.copy()
    df2["Amount (₹)"] = pd.to_numeric(df2["Amount (₹)"], errors="coerce")
    df2 = df2.dropna(subset=["Amount (₹)"])
    if df2.empty:
        return {}

    # ---- 1) Ensure date data is present (use synthetic if needed) ----
    # (We use the same logic from compute_spending_trends to ensure it works)
    has_date = False
    for c in ["Date", "date", "Transaction Date", "Txn Date"]:
        if c in df2.columns:
            df2["__date"] = pd.to_datetime(df2[c], errors="coerce", dayfirst=True)
            has_date = True
            break

    if not has_date:
        # synthetic
        n = len(df2)
        end_date = pd.Timestamp.today().normalize()
        start_date = end_date - pd.Timedelta(days=n - 1)
        df2["__date"] = pd.date_range(start=start_date, end=end_date, periods=n)

    df2 = df2.sort_values("__date")

    # ---- 2) Monthly spending trend ----
    df2["Month"] = df2["__date"].dt.to_period("M").astype(str)
    monthly = df2.groupby("Month")["Amount (₹)"].sum().reset_index()

    if len(monthly) < 2:
        # Not enough data for a trend → fallback: just use last month as prediction
        last_month = float(monthly["Amount (₹)"].iloc[-1])
        return {
            "next_month_spend": last_month,
            "trend_strength": "unknown",
            "category_drift": {},
            "risk_categories": [],
        }

    # simple linear prediction (slope): month index → spending
    monthly["idx"] = range(len(monthly))
    slope = (
        monthly["Amount (₹)"].iloc[-1] - monthly["Amount (₹)"].iloc[0]
    ) / max(1, len(monthly) - 1)

    next_month_spend = float(monthly["Amount (₹)"].iloc[-1] + slope)

    # Trend strength label
    if slope > 1000:
        trend_strength = "Strong Upward Spending Trend"
    elif slope > 0:
        trend_strength = "Mild Upward Trend"
    elif slope < -1000:
        trend_strength = "Strong Downward Spending Trend"
    else:
        trend_strength = "Mild Downward/Stabilizing Trend"

    # ---- 3) Predict category drift (like mini ML) ----
    cat_month = (
        df2.groupby(["Month", "Category"])["Amount (₹)"].sum().reset_index()
    )

    category_drift = {}
    for cat in cat_month["Category"].unique():
        sub = cat_month[cat_month["Category"] == cat].sort_values("Month")
        if len(sub) < 2:
            continue

        # simple slope on monthly values
        slope_cat = (sub["Amount (₹)"].iloc[-1] - sub["Amount (₹)"].iloc[0]) / max(
            1, len(sub) - 1
        )
        last_val = sub["Amount (₹)"].iloc[-1]
        predicted_next = last_val + slope_cat

        pct_change = (
            (predicted_next - last_val) / last_val * 100 if last_val > 0 else 0
        )

        category_drift[cat] = round(pct_change, 1)

    # ---- 4) Identify high-risk categories (likely to overspend) ----
    risk_categories = [
        cat for cat, pct in category_drift.items() if pct >= 20
    ]

    return {
        "next_month_spend": next_month_spend,
        "trend_strength": trend_strength,
        "category_drift": category_drift,
        "risk_categories": risk_categories,
    }





def build_personalized_recommendations(
    prediction,
    df: pd.DataFrame,
    savings_capacity: float,
    goals_df: pd.DataFrame | None,
    selected_gurus: list | None = None,
):
    """
    Generates personalized action items based on:
      - predictive model (spend trend, category drift)
      - goals
      - savings ability
      - guru preferences
    """

    if not prediction:
        return "No prediction data available."

    recs = []
    drift = prediction.get("category_drift", {})
    risk_cats = prediction.get("risk_categories", [])

    # ---- 1) Over-spending risk ----
    if prediction["trend_strength"].startswith("Strong Upward"):
        recs.append("• Your spending trend is rising fast — consider tightening Wants for 2–3 weeks.")

    for cat in risk_cats:
        recs.append(f"• **{cat}** is projected to spike next month — set a soft cap and monitor daily.")

    # ---- 2) Savings & goals ----
    if savings_capacity < 2000:
        recs.append("• Your saving capacity is low — freeze big purchases for a short period.")
    else:
        recs.append(f"• You can save approx **₹{savings_capacity:,.0f}**/month — allocate using a SIP or auto-debit.")

    if goals_df is not None and not goals_df.empty:
        high_priority_goals = goals_df[goals_df["Priority (1-5)"] >= 4]
        if not high_priority_goals.empty:
            for _, row in high_priority_goals.iterrows():
                recs.append(
                    f"• Boost monthly contribution to **{row['Goal Name']}** to stay on track."
                )

    # ---- 3) Guru-based personalized advice ----
    if selected_gurus:
        if "Warren Buffett" in selected_gurus:
            recs.append("• Buffett principle: Avoid lifestyle inflation — lock your spending baseline now.")
        if "Robert Kiyosaki" in selected_gurus:
            recs.append("• Kiyosaki tip: Channel more money into assets — try a recurring mutual fund SIP.")
        if "Ramit Sethi" in selected_gurus:
            recs.append("• Ramit Sethi rule: Spend more on what you love, cut relentlessly on what you don’t.")
        if "Indian Tax Guru" in selected_gurus:
            recs.append("• Tax Guru: Maximize deductions with PPF / ELSS / NPS if not already done.")

    # ---- FINAL: Format as bullet points ----
    if recs:
        return "\n".join(recs)
    return "No strong recommendations — keep up your current pattern!"






# ---------- Helper: Generate financial advice ----------
# ---------- FINANCIAL ADVICE TOOL (guru-aware) ----------

def generate_financial_advice(
    df: pd.DataFrame,
    selected_gurus,
    guru_notes_text: str,
    total_spent: float,
    guru_analysis: dict | None = None,
):
    """
    Build structured advice using:
      - spending by category
      - chosen guru philosophies
      - uploaded guru documents (multiple books/articles)
      - otherwise a default 'best book' style profile
    """
    if df.empty:
        return "No expenses detected to analyze."

    # Spend per category
    cat_totals = df.groupby("Category")["Amount (₹)"].sum().to_dict()

    lines = []
    lines.append(f"💵 You spent a total of **₹{total_spent:.2f}** in the documents you uploaded.")

    # Category breakdown
    if cat_totals:
        top_cat = max(cat_totals, key=cat_totals.get)
        lines.append(
            "📊 Your highest spending category is **{cat}** with **₹{amt:.0f}**.".format(
                cat=top_cat, amt=cat_totals[top_cat]
            )
        )

    # Decide whether uploaded content is useful
    use_uploaded = bool(guru_analysis and guru_analysis.get("is_useful", False))

    # Fallback if user did not select any guru
    if not selected_gurus:
        selected_gurus = ["Warren Buffett"]

    lines.append("")
    lines.append("🧠 **Guru-based insights:**")

    # Core guru-style advice
    for guru in selected_gurus:
        profile = GURU_PROFILES.get(guru)
        if not profile:
            continue

        lines.append(f"\n**{guru} style:** {profile['summary']}")

        focus = profile.get("focus_categories", [])
        high_focus = [c for c in focus if c in cat_totals and cat_totals[c] > 0]
        if high_focus:
            focus_str = ", ".join(high_focus)
            lines.append(
                f"• Your spending in **{focus_str}** is especially relevant to {guru}'s philosophy."
            )

        for rule in profile["core_rules"][:2]:
            lines.append(f"• {rule}")

    # ---------- Book-based section ----------
    lines.append("\n📚 **Book-based recommendations:**")

    if use_uploaded:
        # Use the uploaded financial content (multi-source)
        wc = guru_analysis.get("word_count", 0)
        pages = guru_analysis.get("approx_pages", 0)
        themes = guru_analysis.get("themes", [])
        primary_src = guru_analysis.get("primary_source")
        sources = guru_analysis.get("sources", []) or []

        if primary_src:
            lines.append(
                f"- Using your uploaded financial content, mainly **{primary_src}**, "
                f"plus {max(0, len(sources)-1)} other file(s) as reference."
            )
        else:
            lines.append(
                "- Using your uploaded financial books/articles as the main reference."
            )

        if pages and wc:
            lines.append(
                f"- Total analyzed content ≈ **{wc} words** (~**{pages}** pages)."
            )

        if themes:
            lines.append(
                "- The text strongly focuses on: " +
                ", ".join(f"**{t}**" for t in themes) + "."
            )

        lines.append(
            "- Over the next month, try to align your highest spending categories with these themes. "
            "For example, if the book emphasizes **Budgeting & Expense Control**, focus on cutting overspending "
            "in Food/Shopping/Entertainment first."
        )

    else:
        # No strong uploaded financial content → choose best online-style book
        # This is your "intelligent default book picker"
        book_name, book_profile = choose_default_book(selected_gurus, guru_analysis or {})

        lines.append(
            f"- No strong financial book/article detected, so we are using "
            f"**{book_name}** as a reference (picked as the best fit for your gurus/themes)."
        )

        lines.append(f"- Summary: {book_profile['summary']}")

        # DEFAULT_BOOK_PROFILES use 'key_ideas'
        key_ideas = book_profile.get("key_ideas") or book_profile.get("core_rules") or []
        for idea in key_ideas:
            lines.append(f"• {idea}")

        lines.append(
            "- Try to apply at least **one idea** from this book to your top spending category this week."
        )

    # Simple 50–30–20 style reminder
    lines.append(
        "\n📌 Try using a **50–30–20 style plan**: "
        "around 50% on needs (Food, Bills, Groceries), "
        "30% on wants (Entertainment, Shopping, some Transport), "
        "and 20% for savings/investments."
    )

    return "\n".join(lines)



def build_multi_guru_comparison(
    df: pd.DataFrame,
    selected_gurus,
    guru_analysis: dict | None = None,
):
    """
    Build a comparison of how your spending lines up with each selected guru's philosophy.

    Returns:
      - summary_md: markdown text
      - comp_df: DataFrame with per-guru stats
    """
    if df is None or df.empty:
        return "No expenses available to compare with gurus.", pd.DataFrame()

    if not selected_gurus:
        return "No gurus selected. Please select at least one guru above.", pd.DataFrame()

    df2 = df.copy()
    df2["Amount (₹)"] = pd.to_numeric(df2["Amount (₹)"], errors="coerce")
    df2 = df2.dropna(subset=["Amount (₹)"])
    if df2.empty:
        return "Could not compute comparison because amounts are missing.", pd.DataFrame()

    total_spent = float(df2["Amount (₹)"].sum())
    cat_totals = df2.groupby("Category")["Amount (₹)"].sum().to_dict()

    themes = []
    if guru_analysis and isinstance(guru_analysis, dict):
        themes = guru_analysis.get("themes") or []
    themes = [t for t in themes if t]

    rows = []
    lines = []
    lines.append("Here’s how your current spending lines up with each selected guru’s core focus:")

    for guru in selected_gurus:
        profile = GURU_PROFILES.get(guru)
        if not profile:
            continue

        focus_cats = profile.get("focus_categories", []) or []
        # spending in focus categories
        focus_spend = sum(cat_totals.get(c, 0.0) for c in focus_cats)
        focus_pct = (focus_spend / total_spent * 100.0) if total_spent > 0 else 0.0

        # short description
        summary = profile.get("summary", "").strip()
        if len(summary) > 140:
            summary_short = summary[:137].rstrip() + "..."
        else:
            summary_short = summary

        # Very simple "alignment" label
        if focus_pct >= 40:
            alignment = "Strong alignment – large share of your spending is in their focus zones."
        elif focus_pct >= 20:
            alignment = "Partial alignment – some of your spending is in their focus zones."
        else:
            alignment = "Weak alignment – relatively little of your spending is in their focus zones."

        # Theme mention if we have any
        theme_note = ""
        if themes:
            theme_note = "Themes in your uploaded books/articles: " + ", ".join(themes)

        rows.append(
            {
                "Guru": guru,
                "Focus Categories": ", ".join(focus_cats) if focus_cats else "Not specified",
                "Spending in Focus (₹)": focus_spend,
                "Focus Share of Total (%)": round(focus_pct, 1),
                "Alignment": alignment,
            }
        )

        lines.append(f"\n### {guru}")
        lines.append(f"- Style: {summary_short}")
        if focus_cats:
            focus_str = ", ".join(focus_cats)
            lines.append(f"- Focus areas: **{focus_str}**")
        lines.append(
            f"- You currently spend about **₹{focus_spend:,.0f}** here "
            f"(≈ **{focus_pct:.1f}%** of all tracked expenses)."
        )
        lines.append(f"- Alignment: {alignment}")
        if theme_note:
            lines.append(f"- {theme_note}")

    if not rows:
        return "No valid guru profiles loaded to compare.", pd.DataFrame()

    comp_df = pd.DataFrame(rows).sort_values(
        by="Focus Share of Total (%)", ascending=False
    )

    summary_md = "\n".join(lines)
    return summary_md, comp_df





def generate_indian_investment_advice(
    df: pd.DataFrame,
    monthly_income: float,
    target_saving_pct: float,
):
    """
    India-specific financial advice:
      - PPF / ELSS / SIP suggestions
      - Typical rent/EMI/lifestyle vs income guidance
    Returns markdown string.
    """
    if df is None or df.empty or monthly_income <= 0:
        return (
            "Not enough data or income not set to provide India-specific investment advice. "
            "Please enter your monthly income and upload some expenses."
        )

    lines = []

    total_spent = float(df["Amount (₹)"].sum())
    planned_saving = monthly_income * (target_saving_pct / 100.0)
    actual_saving = max(monthly_income - total_spent, 0.0)
    savings_gap = planned_saving - actual_saving

    by_cat = df.groupby("Category")["Amount (₹)"].sum().to_dict()

    rent_spend = by_cat.get("Rent", 0.0)
    emi_spend = by_cat.get("Loans & EMIs", 0.0)
    invest_spend = by_cat.get("Investments", 0.0)
    food_spend = by_cat.get("Food", 0.0)
    shop_spend = by_cat.get("Shopping", 0.0)
    ent_spend = by_cat.get("Entertainment", 0.0)
    transport_spend = by_cat.get("Transport", 0.0)

    rent_ratio = rent_spend / monthly_income if monthly_income > 0 else 0
    emi_ratio = emi_spend / monthly_income if monthly_income > 0 else 0
    invest_ratio = invest_spend / monthly_income if monthly_income > 0 else 0

    lines.append(
        f"💼 Based on a monthly income of **₹{format_inr(monthly_income, 0)}** and target savings of "
        f"**{target_saving_pct:.0f}%**, here is India-specific guidance for you:"
    )

    # ---------- 1) Savings & SIP range ----------
    rec_sip_low = monthly_income * 0.10   # 10%
    rec_sip_high = monthly_income * 0.20  # 20%

    lines.append(
        f"\n🪙 **SIP / Mutual Fund Investing**\n"
        f"- For most Indian salaried individuals, a good thumb rule is to invest **10–20% of income via SIPs**.\n"
        f"- For you, that means a monthly SIP range of roughly **₹{rec_sip_low:,.0f} – ₹{rec_sip_high:,.0f}**."
    )

    if invest_spend > 0:
        lines.append(
            f"- You are currently tagging around **₹{invest_spend:,.0f} per month** as `Investments` "
            f"(≈ **{invest_ratio*100:.1f}%** of income)."
        )
        if invest_ratio < 0.08:
            lines.append(
                "- This is on the **lower side**. Try to gradually increase your SIP amount towards the 10–15% mark."
            )
        elif invest_ratio <= 0.20:
            lines.append(
                "- This is a **healthy investment rate**. Staying consistent for years will build strong wealth."
            )
        else:
            lines.append(
                "- You are investing an **aggressive** share of your income. Ensure your emergency fund is strong "
                "and EMIs/essentials are fully comfortable."
            )
    else:
        lines.append(
            "- It looks like you have **no explicit `Investments` category** yet. Consider starting at least one SIP "
            "with a small amount and increasing it over time."
        )

    # ---------- 2) Rent & EMI checks ----------
    lines.append("\n🏠 **Rent & EMI Check (Indian context)**")

    if rent_spend > 0:
        lines.append(
            f"- Your tracked **rent/PG/hostel** spend is about **₹{rent_spend:,.0f}** "
            f"(≈ **{rent_ratio*100:.1f}%** of income)."
        )
        if rent_ratio > 0.4:
            lines.append(
                "- This is quite **high**. Ideally, house rent should be within ~25–35% of income. "
                "Consider renegotiating, finding cheaper accommodation, or sharing to reduce pressure."
            )
        elif rent_ratio > 0.3:
            lines.append(
                "- This is on the **upper side** of comfortable. Keep other fixed costs (EMIs, loans) under control."
            )
        else:
            lines.append(
                "- Your rent level looks **reasonable** relative to your income."
            )
    else:
        lines.append("- No explicit `Rent` category detected. If you pay rent, consider tagging it so analysis is accurate.")

    if emi_spend > 0:
        lines.append(
            f"- Your **Loans & EMIs** total to about **₹{emi_spend:,.0f}** (≈ **{emi_ratio*100:.1f}%** of income)."
        )
        if emi_ratio > 0.4:
            lines.append(
                "- EMIs above **40% of income** put a lot of strain on cash flow. Avoid taking new loans and "
                "plan to close high-interest ones first."
            )
        elif emi_ratio > 0.25:
            lines.append(
                "- EMIs are **moderate to high**. Try not to add new EMIs and build an emergency fund alongside."
            )
        else:
            lines.append(
                "- Your EMI levels look **manageable**. Focus on steady investing and prepaying expensive loans when possible."
            )
    else:
        lines.append("- No `Loans & EMIs` found in your data. If you have loans, tag EMI payments for better tracking.")

    # ---------- 3) Lifestyle spending vs freeing money for SIP/PPF/ELSS ----------
    lifestyle_total = food_spend + shop_spend + ent_spend + transport_spend
    if total_spent > 0:
        lifestyle_ratio = lifestyle_total / total_spent
    else:
        lifestyle_ratio = 0

    lines.append("\n🍛 **Lifestyle vs Wealth-Building**")

    lines.append(
        f"- Food outside, Shopping, Entertainment and Transport together are about "
        f"**₹{lifestyle_total:,.0f}** of spending "
        f"(≈ **{lifestyle_ratio*100:.1f}%** of your total tracked expenses)."
    )

    if lifestyle_ratio > 0.5:
        lines.append(
            "- This is a **very high lifestyle share**. Even small cuts here (eating out less, fewer random purchases) "
            "can free money for SIPs, PPF, or loan prepayment."
        )
    elif lifestyle_ratio > 0.35:
        lines.append(
            "- Lifestyle spending is **moderate**. You can still find room for optimization if savings feel tight."
        )
    else:
        lines.append(
            "- Lifestyle spending is **quite controlled**. Good base for increasing investments."
        )

    # ---------- 4) PPF, ELSS, 80C style guidance ----------
    lines.append("\n📘 **PPF / ELSS / Tax-Saving Basics (India)**")
    lines.append(
        "- Under Section **80C**, you can invest up to **₹1.5 lakh per financial year** in instruments like **PPF, ELSS, "
        "EPF, life insurance premiums, etc.** and get tax benefits."
    )
    lines.append(
        "- **PPF**: 15-year lock-in, government-backed, stable interest, good for safe long-term corpus."
    )
    lines.append(
        "- **ELSS mutual funds**: Market-linked with a 3-year lock-in. Suitable if you can take some equity risk and want tax-saving + growth."
    )
    lines.append(
        "- A common approach is to split your investment between **SIP in equity funds (for growth)** and **PPF/FD/RD (for stability)** "
        "based on your risk comfort."
    )

    # Suggest a simple split for them
    safe_part = planned_saving * 0.4 if planned_saving > 0 else monthly_income * 0.05
    growth_part = planned_saving * 0.6 if planned_saving > 0 else monthly_income * 0.10

    lines.append(
        f"- Given your target savings, one simple plan could be:\n"
        f"  • Around **₹{growth_part:,.0f} per month** into **SIPs / ELSS / equity mutual funds** (growth part)\n"
        f"  • Around **₹{safe_part:,.0f} per month** into **PPF / RD / safer options** (stability part)\n"
        "  Adjust these numbers slowly over time as income and comfort grow."
    )

    # ---------- 5) If savings gap exists ----------
    if savings_gap > 0:
        lines.append(
            f"\n⚠️ You are currently about **₹{savings_gap:,.0f} below** your target savings based on this data. "
            "Try moving a small part of Food/Shopping/Entertainment spend into a fixed SIP every month."
        )
    else:
        lines.append(
            "\n✅ Your data suggests you can **hit or exceed your savings target** if this pattern continues. "
            "Focus on consistency and avoid lifestyle inflation when income rises."
        )

    return "\n".join(lines)




def generate_indian_tax_saving_recommendations(
    df: pd.DataFrame,
    monthly_income: float,
    target_saving_pct: float,
):
    """
    India-focused tax-saving tips using common sections:
      - 80C (PPF, ELSS, EPF, home loan principal, tuition)
      - 80D (health insurance)
      - 80CCD(1B) (NPS extra)
    This is educational only – not professional tax advice.
    """
    if df is None or df.empty or monthly_income <= 0:
        return (
            "To see detailed tax-saving suggestions, please enter your monthly income "
            "and upload some expenses including Investments / Health / Loans if possible.\n\n"
            "_Disclaimer: This is educational guidance only. For exact tax calculation, consult a CA or tax professional._"
        )

    annual_income = monthly_income * 12
    by_cat = df.groupby("Category")["Amount (₹)"].sum().to_dict()

    invest_spend = by_cat.get("Investments", 0.0)      # SIPs, mutual funds, PPF etc.
    health_spend = by_cat.get("Health", 0.0)           # Doctor+medicine, not exact 80D but indicative
    emi_spend = by_cat.get("Loans & EMIs", 0.0)

    # Rough annualized values
    annual_invest = invest_spend * 12
    annual_health = health_spend * 12

    # Common Indian limits (old regime style reference)
    limit_80C = 150000  # ₹1.5 lakh
    limit_80D_base = 25000  # ₹25k for self/family (non-senior)
    limit_80CCD1B = 50000   # ₹50k extra for NPS

    lines = []
    lines.append("📜 **India – Tax-Saving Opportunities (Educational Overview)**")
    lines.append(
        "_These pointers are based on common Indian tax sections (80C, 80D, 80CCD(1B)) and are for education only. "
        "Actual eligibility depends on the exact products you use and your chosen tax regime._"
    )

    lines.append(
        f"\n💼 Approximate annual income considered: **{format_inr(annual_income, 0)}**"
    )

    # ---------- 80C style analysis ----------
    lines.append("\n🔹 **Section 80C (up to ₹1.5 lakh per year)**")
    lines.append(
        "Eligible instruments typically include PPF, ELSS mutual funds, EPF, certain life insurance premiums, "
        "home loan principal repayment, and some tuition fees for children."
    )

    lines.append(
        f"- Based on your `Investments` category, you are roughly investing **{format_inr(annual_invest, 0)} per year**."
    )

    gap_80c = max(limit_80C - annual_invest, 0)
    if gap_80c > 0:
        lines.append(
            f"- You may still have up to about **{format_inr(gap_80c, 0)}** of unused 80C limit "
            "if these investments are in 80C-eligible products (PPF, ELSS, etc.)."
        )
        lines.append(
            "- If your income and risk profile allow, you can consider increasing SIPs into ELSS / tax-saving mutual funds, "
            "or adding PPF/EPF contributions to utilize more of this limit."
        )
    else:
        lines.append(
            "- You seem to be **near or above** the standard 80C limit already based on your current Investment trend. "
            "Make sure you are not double-counting and that these products actually qualify under 80C."
        )

    # ---------- 80D style guidance ----------
    lines.append("\n🔹 **Section 80D (Health Insurance Premiums)**")
    lines.append(
        "Premiums paid for health insurance (for self, spouse, children, and parents) can usually be claimed under 80D. "
        "The common limit is around **₹25k** for self/family and additional for parents (especially if senior citizens)."
    )

    if annual_health > 0:
        lines.append(
            f"- Your `Health` category spending is around **{format_inr(annual_health, 0)} per year** "
            "(this may include medicines + doctor visits, not just insurance)."
        )
    else:
        lines.append(
            "- No major `Health` spending is detected. If you do have a health insurance policy, consider tracking those premiums separately."
        )

    lines.append(
        "- For better protection and tax benefit, ensure you have a **proper health insurance policy**; "
        "premiums typically qualify for 80D up to the specified limits."
    )

    # ---------- 80CCD(1B) NPS style guidance ----------
    lines.append("\n🔹 **Section 80CCD(1B) – NPS (up to ₹50k extra)**")
    lines.append(
        "If you invest in the **National Pension System (NPS)**, contributions over and above 80C (up to **₹50,000** per year) "
        "may be claimed under 80CCD(1B)."
    )
    lines.append(
        "- If your 80C limit is already nearly full and you want additional tax-saving with retirement focus, "
        "consider a small monthly NPS contribution that fits your risk and liquidity comfort."
    )

    # ---------- Simple numeric suggestion ----------
    planned_saving = monthly_income * (target_saving_pct / 100.0)
    lines.append("\n📊 **Simple Suggested Structure (Educational)**")

    if planned_saving > 0:
        approx_80c_monthly = min(limit_80C, planned_saving * 12) / 12
        approx_nps_monthly = limit_80CCD1B / 12

        lines.append(
            f"- To gradually move towards using 80C, you could target around **{format_inr(approx_80c_monthly, 0)} per month** "
            "in 80C-eligible investments (PPF, ELSS, EPF, etc.), depending on your existing contributions."
        )
        lines.append(
            f"- If suitable, an additional **{format_inr(approx_nps_monthly, 0)} per month** into NPS could help you "
            "utilize the 80CCD(1B) deduction over the year."
        )
    else:
        lines.append(
            "- Once you define a clear savings target, you can split it between tax-saving options like 80C products and NPS."
        )

    # ---------- Final disclaimer ----------
    lines.append(
        "\n⚠️ **Important Disclaimer:**\n"
        "- Actual tax rules can change and depend on your chosen tax regime (old vs new), age, and specific products.\n"
        "- Use this as a **starting blueprint** and always verify with the latest rules or consult a CA/tax professional "
        "before making major decisions."
    )

    return "\n".join(lines)






# ---------- ADVANCED OCR HELPERS (multi-format + PDF) ----------

# If you installed poppler, set the bin path here.
# If you don't want to use poppler yet, set POPPLER_PATH = None
POPPLER_PATH = r"C:\poppler\Library\bin"  # change if your poppler is elsewhere, or set to None

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "tiff", "bmp", "pdf"]


def extract_text_from_image(image):
    """Run Tesseract OCR on a single image."""
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(file_bytes):
    """
    1) Try PyPDF2 for normal text PDFs (most bank statements).
    2) If that fails or returns empty, fall back to pdf2image + Tesseract (needs Poppler).
    Returns: (text, preview_image or None)
    """
    # ---- 1) Try direct text extraction (no Poppler needed) ----
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            texts.append(f"\n\n===== PAGE {i} (text) =====\n\n{page_text}")
        full_text = "".join(texts).strip()
        if full_text:
            # We don't have a preview image here, but that's fine
            return full_text, None
    except Exception:
        # Ignore and fall back to image-based OCR
        pass

    # ---- 2) Fallback: convert pages to images + Tesseract ----
    kwargs = {}
    if POPPLER_PATH and os.path.isdir(POPPLER_PATH):
        kwargs["poppler_path"] = POPPLER_PATH

    try:
        pages = convert_from_bytes(file_bytes, dpi=300, **kwargs)
    except Exception as e:
        # Give a clearer error instead of "Unable to get page count"
        raise RuntimeError(
            f"PDF OCR failed. Likely Poppler is missing or POPPLER_PATH is wrong. Details: {e}"
        )

    all_text = []
    for i, page in enumerate(pages, start=1):
        page_text = pytesseract.image_to_string(page)
        all_text.append(f"\n\n===== PAGE {i} (image OCR) =====\n\n{page_text}")

    preview = pages[0] if pages else None
    return "\n".join(all_text), preview


def advanced_ocr(uploaded_file):
    """
    Detect file type (image / pdf) and route to correct OCR pipeline.
    Returns (text, preview_image)
    """
    file_bytes = uploaded_file.getvalue()
    filename = (uploaded_file.name or "").lower()

    is_pdf = filename.endswith(".pdf") or uploaded_file.type == "application/pdf"

    if is_pdf:
        return extract_text_from_pdf(file_bytes)
    else:
        image = Image.open(io.BytesIO(file_bytes))
        text = extract_text_from_image(image)
        return text, image





# ---------- SPLITWISE CSV INTEGRATION ----------

def parse_splitwise_csv(uploaded_file):
    """
    Parse a Splitwise-like CSV export into our standard format:
    Item, Amount (₹), Category, Group, Person

    This version is more flexible and tries to auto-detect the right columns.
    """
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read Splitwise CSV: {e}")
        return pd.DataFrame()

    # Show columns once for debugging
    st.info("Splitwise CSV columns detected: " + ", ".join(list(df_raw.columns)))

    # Lowercase map for matching
    colmap = {c.lower().strip(): c for c in df_raw.columns}

    def pick(*names):
        """Try to pick a column by a set of possible lowercase names."""
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    # Try to find description-like column
    desc_col = pick(
        "description",
        "details",
        "expense",
        "narration",
        "note",
        "comments",
    )

    # Try to find numeric "your share" / amount column
    amount_col = pick(
        "your share",
        "owed share",
        "amount you owe",
        "amount",
        "cost",
        "value",
    )

    # Fallback: if still not found, guess…
    if desc_col is None and len(df_raw.columns) >= 1:
        # pick first non-numeric-looking column as description
        for c in df_raw.columns:
            if not pd.api.types.is_numeric_dtype(df_raw[c]):
                desc_col = c
                break

    if amount_col is None:
        # pick first numeric column as amount
        for c in df_raw.columns:
            if pd.api.types.is_numeric_dtype(df_raw[c]):
                amount_col = c
                break

    # If even after fallback nothing is found, give clear message
    if not desc_col or not amount_col:
        st.error(
            "Splitwise CSV: could not find a description/amount column.\n"
            "Please check the CSV headers or export again from Splitwise."
        )
        return pd.DataFrame()

    cat_col = pick("category")
    group_col = pick("group", "group name")
    person_col = pick("friend", "with you", "paid by", "payer", "person")

    expenses = []

    for _, row in df_raw.iterrows():
        # Amount
        raw_amt = row[amount_col]
        try:
            amt = float(str(raw_amt).replace(",", "").strip())
        except Exception:
            continue
        if amt <= 0:
            continue

        # Description
        desc = row[desc_col]
        desc_str = "Splitwise Expense" if pd.isna(desc) else str(desc).strip()
        if not desc_str:
            desc_str = "Splitwise Expense"

        # Category: from CSV if exists, else detect
        if cat_col and not pd.isna(row[cat_col]):
            cat = str(row[cat_col]).strip() or "Others"
        else:
            cat = detect_category(desc_str)

        record = {
            "Item": desc_str,
            "Amount (₹)": amt,
            "Category": cat,
        }

        if group_col and not pd.isna(row[group_col]):
            record["Group"] = str(row[group_col]).strip()
        if person_col and not pd.isna(row[person_col]):
            record["Person"] = str(row[person_col]).strip()

        expenses.append(record)

    if not expenses:
        st.warning("Splitwise CSV did not contain usable expenses (all amounts were zero or invalid).")
        return pd.DataFrame()

    return pd.DataFrame(expenses)




# ---------- BUDGET RECOMMENDATION ENGINE ----------

def generate_budget_recommendations(
    df: pd.DataFrame,
    budget_compare_df: pd.DataFrame,
    monthly_income: float,
    target_saving_pct: float,
    bucket_budget: dict,
):
    """
    Create simple, rule-based budgeting recommendations
    using spending patterns and the user's income/savings target.
    Returns a markdown string.
    """
    if df.empty:
        return "No expenses to analyze yet. Upload some data to see recommendations."

    lines = []

    total_spent = float(df["Amount (₹)"].sum())
    planned_saving = monthly_income * (target_saving_pct / 100.0)
    actual_saving = max(monthly_income - total_spent, 0.0)
    savings_gap = planned_saving - actual_saving

    lines.append(
        f"💵 **Total spending analyzed:** ₹{total_spent:.0f} out of a monthly income of ₹{format_inr(monthly_income, 0)}."
    )
    lines.append(
        f"🏦 **Planned savings:** ₹{planned_saving:.0f} ({target_saving_pct:.0f}% of income)."
    )
    lines.append(
        f"📉 **Estimated actual savings from this data:** ₹{actual_saving:.0f}."
    )

    # 1) Savings gap
    if savings_gap > 0:
        lines.append(
            f"⚠️ You are about **₹{savings_gap:.0f} below** your target savings. "
            "You need to reduce expenses by this amount to hit your goal."
        )
    else:
        lines.append(
            "✅ You are currently on track to meet or exceed your savings target based on these expenses."
        )

    # 2) Overspending categories
    if budget_compare_df is not None and not budget_compare_df.empty:
        overs = budget_compare_df[budget_compare_df["Status"] == "High"]
        near = budget_compare_df[budget_compare_df["Status"] == "Near Limit"]

        if not overs.empty:
            overs_list = [
                f"{row['Category']} (spent ₹{row['Spent']:.0f} vs budget ₹{row['Budget']:.0f})"
                for _, row in overs.iterrows()
            ]
            lines.append(
                "\n🔥 **High overspending detected in:** " + "; ".join(overs_list) + "."
            )
            lines.append(
                "👉 For these categories, try setting hard weekly limits or switching some purchases to cheaper alternatives."
            )

        if not near.empty:
            near_list = [
                f"{row['Category']} (₹{row['Spent']:.0f} / ₹{row['Budget']:.0f})"
                for _, row in near.iterrows()
            ]
            lines.append(
                "\n🟡 **Close to budget limit in:** " + "; ".join(near_list) + "."
            )
            lines.append(
                "👉 Be cautious with new spending here; a few more transactions will push these over budget."
            )

    # 3) Needs vs Wants
    df_bucket = df.copy()
    df_bucket["Bucket"] = df_bucket["Category"].map(CATEGORY_TO_BUCKET)
    bucket_spent = df_bucket.groupby("Bucket")["Amount (₹)"].sum().to_dict()

    needs_spent = float(bucket_spent.get("Needs", 0.0))
    wants_spent = float(bucket_spent.get("Wants", 0.0))

    needs_budget = float(bucket_budget.get("Needs", 0.0))
    wants_budget = float(bucket_budget.get("Wants", 0.0))
    savings_budget = float(bucket_budget.get("Savings", 0.0))

    lines.append(
        "\n📊 **Needs vs Wants check:** "
        f"Needs spent ≈ ₹{needs_spent:.0f} (budget ₹{needs_budget:.0f}), "
        f"Wants spent ≈ ₹{wants_spent:.0f} (budget ₹{wants_budget:.0f}), "
        f"Savings target ≈ ₹{savings_budget:.0f}."
    )

    if wants_spent > wants_budget * 1.1:
        lines.append(
            "🎯 Your **Wants** spending is clearly higher than planned. Focus on cutting back "
            "in Entertainment, Shopping, and non-essential Food/Transport for the next few weeks."
        )
    elif needs_spent > needs_budget * 1.1 and wants_spent <= wants_budget:
        lines.append(
            "📌 Most of the pressure is coming from **Needs** (bills, groceries, essentials). "
            "Look for optimizations like cheaper plans, bulk buys, or sharing subscriptions."
        )
    else:
        lines.append(
            "✅ Your Needs vs Wants balance looks reasonable for now. Keeping consistency here will help long-term."
        )

    # 4) Simple next steps
    lines.append(
        "\n📝 **Suggested next steps:**\n"
        "- Set a weekly spending cap for your top 1–2 overspending categories.\n"
        "- Automate transfers for savings at the start of the month (so you spend what’s left).\n"
        "- Review any subscriptions or recurring payments you don’t really use.\n"
        "- Re-run this analysis after 2–3 weeks to see if your pattern improves."
    )

    return "\n\n".join(lines)




def compute_financial_health_score(
    df: pd.DataFrame,
    monthly_income: float,
    target_saving_pct: float,
    bucket_budget: dict,
    budget_compare_df: pd.DataFrame | None,
    goal_df: pd.DataFrame | None,
):
    """
    Build a 0–100 financial health score using:
      - Savings vs target
      - Budget discipline
      - Needs vs Wants balance
      - Goal achievement
      - Anomalies & spikes (from patterns)
    Returns:
      {
        "total": float,
        "grade": str,
        "summary": str,
        "components": [
            {"Area": str, "Score": float, "Max": int, "Comment": str},
            ...
        ],
      }
    """
    if df is None or df.empty or monthly_income <= 0:
        return {
            "total": 0.0,
            "grade": "Needs Attention",
            "summary": "Not enough data or income set to compute a financial health score.",
            "components": [],
        }

    total_spent = float(df["Amount (₹)"].sum())
    planned_saving = monthly_income * (target_saving_pct / 100.0)
    actual_saving = max(monthly_income - total_spent, 0.0)

    def clamp01(x):
        return max(0.0, min(1.0, float(x)))

    components = []

    # ---------- 1) Savings & Spending Balance (0–30) ----------
    max_savings_score = 30
    if planned_saving <= 0:
        savings_score = max_savings_score * 0.5  # neutral if no explicit target
        comment = "No clear savings target set; using a neutral score."
    else:
        ratio = clamp01(actual_saving / planned_saving)
        savings_score = max_savings_score * ratio
        if ratio >= 1.1:
            comment = "You are saving more than your planned target — very strong savings discipline."
        elif ratio >= 0.9:
            comment = "You are roughly on track with your savings target."
        elif ratio >= 0.5:
            comment = "You are below your savings target; some expense tightening can help."
        else:
            comment = "Your savings are far below your target. Spending is consuming most of your income."

    components.append({
        "Area": "Savings & Spending Balance",
        "Score": round(savings_score, 1),
        "Max": max_savings_score,
        "Comment": comment,
    })

    # ---------- 2) Budget Discipline (0–25) ----------
    max_budget_score = 25
    if budget_compare_df is None or budget_compare_df.empty:
        budget_score = max_budget_score * 0.7  # neutral-okay
        comment = "No detailed category budgets available; assuming moderate discipline."
    else:
        total_cats = len(budget_compare_df)
        overs = (budget_compare_df["Status"] == "High").sum()
        near = (budget_compare_df["Status"] == "Near Limit").sum()

        overs_share = overs / total_cats if total_cats > 0 else 0
        near_share = near / total_cats if total_cats > 0 else 0

        raw = 1.0 - (0.7 * overs_share + 0.3 * near_share)
        budget_score = max_budget_score * clamp01(raw)

        if overs_share == 0 and near_share == 0:
            comment = "All categories are within budget — excellent budget discipline."
        elif overs_share <= 0.2 and near_share <= 0.4:
            comment = "A few categories are near/over budget, but overall discipline is okay."
        else:
            comment = "Multiple categories are near or above budget — you may need to re-balance your spending."

    components.append({
        "Area": "Budget Discipline",
        "Score": round(budget_score, 1),
        "Max": max_budget_score,
        "Comment": comment,
    })

    # ---------- 3) Needs vs Wants Mix (0–20) ----------
    max_mix_score = 20

    df_bucket = df.copy()
    df_bucket["Bucket"] = df_bucket["Category"].map(CATEGORY_TO_BUCKET)
    bucket_spent = df_bucket.groupby("Bucket")["Amount (₹)"].sum().to_dict()

    needs_spent = float(bucket_spent.get("Needs", 0.0))
    wants_spent = float(bucket_spent.get("Wants", 0.0))

    actual_total = needs_spent + wants_spent
    if actual_total <= 0:
        mix_score = max_mix_score * 0.6
        comment = "Not enough categorized Needs/Wants data; assuming a neutral mix."
    else:
        actual_needs_ratio = needs_spent / actual_total
        actual_wants_ratio = wants_spent / actual_total

        planned_needs = float(bucket_budget.get("Needs", 0.0))
        planned_wants = float(bucket_budget.get("Wants", 0.0))
        planned_total = planned_needs + planned_wants

        if planned_total <= 0:
            target_needs_ratio = 0.6
            target_wants_ratio = 0.4
        else:
            target_needs_ratio = planned_needs / planned_total
            target_wants_ratio = planned_wants / planned_total

        diff_needs = abs(actual_needs_ratio - target_needs_ratio)
        diff_wants = abs(actual_wants_ratio - target_wants_ratio)
        avg_diff = (diff_needs + diff_wants) / 2.0

        # If avg_diff = 0 → full score; if 0.25+ (very off) → close to 0
        raw = 1.0 - clamp01(avg_diff / 0.25)
        mix_score = max_mix_score * raw

        if avg_diff < 0.05:
            comment = "Your Needs vs Wants balance is very close to the recommended mix."
        elif avg_diff < 0.12:
            comment = "Your Needs vs Wants mix is reasonably balanced with minor drift."
        else:
            comment = "Your Needs vs Wants mix is quite off — either essentials are too high or wants are dominating."

    components.append({
        "Area": "Needs vs Wants Mix",
        "Score": round(mix_score, 1),
        "Max": max_mix_score,
        "Comment": comment,
    })

    # ---------- 4) Goal Achievement (0–15) ----------
    max_goal_score = 15
    if goal_df is None or goal_df.empty:
        goal_score = max_goal_score * 0.6
        comment = "No specific category goals set; score is neutral."
    else:
        total_goals = len(goal_df)
        exceeded = (goal_df["Status"] == "Exceeded").sum()
        near = (goal_df["Status"] == "Near Goal").sum()
        under = (goal_df["Status"] == "Under Goal").sum()

        good_share = under / total_goals if total_goals > 0 else 0
        near_share = near / total_goals if total_goals > 0 else 0
        exceeded_share = exceeded / total_goals if total_goals > 0 else 0

        raw = good_share + 0.5 * near_share - 0.5 * exceeded_share
        goal_score = max_goal_score * clamp01(raw)

        if exceeded_share == 0 and good_share >= 0.7:
            comment = "Most of your category goals are within limits — great progress."
        elif exceeded_share <= 0.3:
            comment = "Some goals are exceeded, but many are still under control."
        else:
            comment = "Several goals are exceeded — you may need to reset or tighten your goals."

    components.append({
        "Area": "Goal Achievement",
        "Score": round(goal_score, 1),
        "Max": max_goal_score,
        "Comment": comment,
    })

    # ---------- 5) Anomalies & Spikes (0–10) ----------
    max_anom_score = 10

    # Reuse the same logic as Tool 5
    pattern_text, daily_spikes_df, anomalies_df = analyze_patterns_and_anomalies(df)

    anomaly_count = len(anomalies_df) if anomalies_df is not None and not anomalies_df.empty else 0
    spike_count = len(daily_spikes_df) if daily_spikes_df is not None and not daily_spikes_df.empty else 0

    # Basic penalty: more anomalies/spikes = lower score
    penalty_raw = anomaly_count / 10.0 + spike_count / 5.0
    raw = 1.0 - clamp01(penalty_raw)
    anom_score = max_anom_score * raw

    if anomaly_count == 0 and spike_count == 0:
        comment = "No major anomalies or spending spikes detected — stable patterns."
    elif anomaly_count <= 3 and spike_count <= 2:
        comment = "A few spikes or unusual transactions — mostly fine but worth a quick review."
    else:
        comment = "Multiple spikes or unusual big transactions — review these for errors or one-off big purchases."

    components.append({
        "Area": "Anomalies & Spikes",
        "Score": round(anom_score, 1),
        "Max": max_anom_score,
        "Comment": comment,
    })

    # ---------- Total & Grade ----------
    total_score = sum(c["Score"] for c in components)
    total_score = max(0.0, min(100.0, total_score))

    if total_score >= 80:
        grade = "Excellent"
        summary = (
            "Your overall financial health looks **strong**. You have good control over spending, "
            "reasonable savings, and mostly stable patterns. Focus on small optimizations."
        )
    elif total_score >= 60:
        grade = "Good"
        summary = (
            "Your financial health is **okay to good**. A few areas (like savings rate, budget discipline, "
            "or anomalies) may need some attention, but overall you're on a decent track."
        )
    elif total_score >= 40:
        grade = "Fair"
        summary = (
            "Your financial health is **under pressure**. Savings, budgets, or spending patterns show stress. "
            "You should prioritize tightening expenses and improving consistency."
        )
    else:
        grade = "Needs Attention"
        summary = (
            "Your financial health is **weak right now**. Savings are low vs income and/or overspending is high. "
            "Consider aggressively cutting wants, setting strict budgets, and re-running this after 2–4 weeks."
        )

    # Optionally highlight weakest area
    weakest = min(components, key=lambda c: c["Score"] / c["Max"] if c["Max"] > 0 else 1)
    weakest_ratio = weakest["Score"] / weakest["Max"] if weakest["Max"] > 0 else 0
    if weakest_ratio < 0.6:
        summary += f"\n\n🔎 The area pulling your score down the most is **{weakest['Area']}**."

    return {
        "total": round(total_score, 1),
        "grade": grade,
        "summary": summary,
        "components": components,
    }



def build_detailed_financial_report(
    df: pd.DataFrame,
    monthly_income: float,
    target_saving_pct: float,
    bucket_budget: dict,
    cat_budget: dict,
    budget_compare_df: pd.DataFrame | None,
    pred_df: pd.DataFrame | None,
):
    """
    Build a multi-section text report summarizing the user's financial picture.
    Returns a markdown string.
    """
    if df is None or df.empty:
        return "No expense data available to generate a report."

    total_spent = float(df["Amount (₹)"].sum())
    by_cat = df.groupby("Category")["Amount (₹)"].sum().sort_values(ascending=False)

    # Reuse your health score engine
    score_result = compute_financial_health_score(
        df=df,
        monthly_income=monthly_income,
        target_saving_pct=target_saving_pct,
        bucket_budget=bucket_budget,
        budget_compare_df=budget_compare_df,
        goal_df=None,  # you can pass real goal_df if you want
    )

    lines = []

    # Header
    lines.append("# Personal Financial Summary Report")
    lines.append("")
    lines.append(f"- Monthly Income (approx): ₹{monthly_income:,.0f}")
    lines.append(f"- Target Savings: {target_saving_pct:.0f}% of income")
    lines.append(f"- Total Spending in this analysis: ₹{total_spent:,.0f}")
    lines.append("")

    # Category overview
    lines.append("## 1. Category-wise Spending Overview")
    lines.append("")
    for cat, amt in by_cat.items():
        lines.append(f"- {cat}: ₹{amt:,.0f}")
    lines.append("")

    # Needs vs Wants
    df_bucket = df.copy()
    df_bucket["Bucket"] = df_bucket["Category"].map(CATEGORY_TO_BUCKET)
    bucket_spent = df_bucket.groupby("Bucket")["Amount (₹)"].sum().to_dict()
    needs_spent = float(bucket_spent.get("Needs", 0.0))
    wants_spent = float(bucket_spent.get("Wants", 0.0))
    others_spent = float(bucket_spent.get("Others", 0.0))
    savings_target = float(bucket_budget.get("Savings", 0.0))

    lines.append("## 2. Needs vs Wants vs Savings")
    lines.append("")
    lines.append(f"- Needs spending (Food, Bills, Rent, etc.): ₹{needs_spent:,.0f}")
    lines.append(f"- Wants spending (Shopping, Entertainment, etc.): ₹{wants_spent:,.0f}")
    lines.append(f"- Other uncategorized spending: ₹{others_spent:,.0f}")
    lines.append(f"- Savings target (per month): ₹{savings_target:,.0f}")
    lines.append("")

    # Budget discipline summary
    lines.append("## 3. Budget Discipline")
    lines.append("")
    if budget_compare_df is None or budget_compare_df.empty:
        lines.append("- No category budgets detected; budget discipline cannot be fully evaluated.")
    else:
        overs = budget_compare_df[budget_compare_df["Status"] == "High"]
        near = budget_compare_df[budget_compare_df["Status"] == "Near Limit"]
        if overs.empty and near.empty:
            lines.append("- All categories are within budget. Excellent budget discipline.")
        else:
            if not overs.empty:
                lines.append("- Categories where spending is **above** budget:")
                for _, row in overs.iterrows():
                    lines.append(
                        f"  - {row['Category']}: spent ₹{row['Spent']:.0f} vs budget ₹{row['Budget']:.0f}"
                    )
            if not near.empty:
                lines.append("- Categories **close to** the budget limit:")
                for _, row in near.iterrows():
                    lines.append(
                        f"  - {row['Category']}: spent ₹{row['Spent']:.0f} vs budget ₹{row['Budget']:.0f}"
                    )
    lines.append("")

    # Prediction summary
    lines.append("## 4. Future Spending Risk (Prediction Summary)")
    lines.append("")
    if pred_df is None or pred_df.empty:
        lines.append("- Not enough data to generate spending predictions.")
    else:
        col_pred = "Predicted Next 30 Days (₹)"
        risky = pred_df[pred_df["Status"].isin(["Likely Over Budget", "Close to Limit"])]
        if risky.empty:
            lines.append("- Current spending pattern does not show major future budget risks.")
        else:
            lines.append("- Categories likely to be under pressure in the next 30 days:")
            for _, row in risky.iterrows():
                lines.append(
                    f"  - {row['Category']}: predicted ≈ ₹{row[col_pred]:.0f} vs budget ₹{row['Budget (₹)']:.0f} "
                    f"({row['Status']})"
                )
    lines.append("")

    # Health score section
    lines.append("## 5. Financial Health Score")
    lines.append("")
    lines.append(f"- Overall score: **{score_result['total']:.0f} / 100**")
    lines.append(f"- Grade: **{score_result['grade']}**")
    lines.append("")
    lines.append(score_result["summary"])
    lines.append("")

    # Final guidance
    lines.append("## 6. Suggested Next Actions")
    lines.append("")
    lines.append("- Focus first on one or two categories where spending is clearly above budget.")
    lines.append("- Try to automate savings/investments at the start of the month to hit your target.")
    lines.append("- Re-run this report after 2–4 weeks to see if your score and patterns improved.")

    return "\n".join(lines)






# ---------- BASIC BUDGETING TOOL (no database) ----------

# Map each category into a bigger bucket (Needs / Wants / Others)
CATEGORY_TO_BUCKET = {
    "Food": "Needs",
    "Groceries": "Needs",
    "Health": "Needs",
    "Bills": "Needs",
    "Rent": "Needs",
    "Education": "Needs",

    "Transport": "Wants",        # You can change to Needs if you prefer
    "Shopping": "Wants",
    "Entertainment": "Wants",

    "Loans & EMIs": "Needs",     # Debt payments are mandatory
    "Investments": "Savings",    # Treated like savings/investing

    "Others": "Others",
}


def compute_budget_allocation(monthly_income: float, target_saving_pct: float = 20.0):
    """
    Use a simple 50-30-20 style rule:
      - Savings  = target_saving_pct of income
      - Remaining money -> half Needs, half Wants
    Return:
      bucket_budget: dict for Needs/Wants/Savings
      cat_budget:    dict for each category (Food, Bills, etc.)
    """
    if monthly_income <= 0:
        return {"Needs": 0, "Wants": 0, "Savings": 0}, {}

    target_saving = monthly_income * (target_saving_pct / 100.0)
    leftover = max(monthly_income - target_saving, 0)

    needs_budget = leftover * 0.5
    wants_budget = leftover * 0.5

    bucket_budget = {
        "Needs": needs_budget,
        "Wants": wants_budget,
        "Savings": target_saving,
    }

    # split Needs and Wants budget equally among their categories
    cat_budget = {}
    needs_cats = [c for c, b in CATEGORY_TO_BUCKET.items() if b == "Needs"]
    wants_cats = [c for c, b in CATEGORY_TO_BUCKET.items() if b == "Wants"]

    for c in needs_cats:
        cat_budget[c] = needs_budget / max(len(needs_cats), 1)
    for c in wants_cats:
        cat_budget[c] = wants_budget / max(len(wants_cats), 1)

    return bucket_budget, cat_budget


def analyze_spending_vs_budget(df: pd.DataFrame, cat_budget: dict):
    """
    Compare actual spending per category vs its budget.
    Returns DataFrame: Category, Spent, Budget, Status
      Status = OK / Near Limit / High
    """
    if df.empty or not cat_budget:
        return pd.DataFrame()

    spent = df.groupby("Category")["Amount (₹)"].sum().reset_index()
    spent.rename(columns={"Amount (₹)": "Spent"}, inplace=True)

    rows = []
    for _, row in spent.iterrows():
        cat = row["Category"]
        spent_amt = float(row["Spent"])
        budget_amt = float(cat_budget.get(cat, 0.0))

        if budget_amt <= 0:
            status = "No Budget"
        elif spent_amt > 1.1 * budget_amt:
            status = "High"
        elif spent_amt > budget_amt:
            status = "Near Limit"
        else:
            status = "OK"

        rows.append(
            {
                "Category": cat,
                "Spent": spent_amt,
                "Budget": budget_amt,
                "Status": status,
            }
        )

    return pd.DataFrame(rows)




def predict_future_expenses(
    df: pd.DataFrame,
    cat_budget: dict,
    horizon_days: int = 30,
):
    """
    Predict future expenses per category for the next `horizon_days` (default 30).

    - If Date column exists:
        * Calculate daily spend per category over observed period
        * Project that rate into future
    - If no Date column:
        * Treat current data as approx one month and use it as prediction.

    Returns DataFrame:
        Category, Observed Period (days), Spent (₹),
        Predicted Next {horizon_days} Days (₹), Budget (₹), Status
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df2 = df.copy()
    df2["Amount (₹)"] = pd.to_numeric(df2["Amount (₹)"], errors="coerce")
    df2 = df2.dropna(subset=["Amount (₹)"])
    if df2.empty:
        return pd.DataFrame()

    observed_days = None

    if "Date" in df2.columns:
        df2["__date"] = pd.to_datetime(df2["Date"], errors="coerce", dayfirst=True)
        df2 = df2.dropna(subset=["__date"])
        if not df2.empty:
            d_min = df2["__date"].min()
            d_max = df2["__date"].max()
            observed_days = max((d_max - d_min).days + 1, 1)

    cat_spent = df2.groupby("Category")["Amount (₹)"].sum().reset_index()
    cat_spent.rename(columns={"Amount (₹)": "Spent (₹)"}, inplace=True)

    rows = []
    for _, row in cat_spent.iterrows():
        cat = row["Category"]
        spent = float(row["Spent (₹)"])

        if observed_days and observed_days > 0:
            daily_rate = spent / observed_days
            predicted = daily_rate * horizon_days
            period_days = observed_days
        else:
            # No dates → assume this is ~1 month pattern
            predicted = spent
            period_days = 30

        budget_amt = float(cat_budget.get(cat, 0.0))

        if budget_amt <= 0:
            status = "No Budget"
        elif predicted > 1.1 * budget_amt:
            status = "Likely Over Budget"
        elif predicted > budget_amt:
            status = "Close to Limit"
        else:
            status = "Within Budget"

        rows.append(
            {
                "Category": cat,
                "Observed Period (days)": period_days,
                "Spent (₹)": spent,
                f"Predicted Next {horizon_days} Days (₹)": predicted,
                "Budget (₹)": budget_amt,
                "Status": status,
            }
        )

    if not rows:
        return pd.DataFrame()

    col_pred = f"Predicted Next {horizon_days} Days (₹)"
    return pd.DataFrame(rows).sort_values(by=col_pred, ascending=False)







def analyze_patterns_and_anomalies(df: pd.DataFrame):
    """
    Advanced expense pattern recognition + anomaly detection.

    Returns:
      - insights_md: markdown text with bullet point insights
      - daily_spikes_df: DataFrame of unusual daily spending spikes (if Date available)
      - anomalies_df: DataFrame of unusual individual transactions
    """
    if df is None or df.empty:
        return "", pd.DataFrame(), pd.DataFrame()

    insights = []

    # Ensure Amount is numeric
    df2 = df.copy()
    df2["Amount (₹)"] = pd.to_numeric(df2["Amount (₹)"], errors="coerce")
    df2 = df2.dropna(subset=["Amount (₹)"])
    if df2.empty:
        return "", pd.DataFrame(), pd.DataFrame()

    total_spent = float(df2["Amount (₹)"].sum())

    # ---------- 1) Category concentration patterns ----------
    cat_sum = df2.groupby("Category")["Amount (₹)"].sum().sort_values(ascending=False)
    if not cat_sum.empty and total_spent > 0:
        cat_share = (cat_sum / total_spent).sort_values(ascending=False)

        top_cat = cat_share.index[0]
        top_share = cat_share.iloc[0] * 100

        if top_share >= 40:
            insights.append(
                f"• **{top_cat}** alone accounts for about **{top_share:.1f}%** of your total spending. "
                "This is a heavy concentration in one category – consider tightening here first."
            )
        elif top_share >= 25:
            insights.append(
                f"• **{top_cat}** is your largest category at around **{top_share:.1f}%** of total spending."
            )

        cum_share = cat_share.cumsum()
        num_for_80 = int((cum_share <= 0.8).sum()) or 1
        top_80_cats = list(cat_share.index[:num_for_80])
        insights.append(
            "• Your spending is mostly driven by: " +
            ", ".join(f"**{c}**" for c in top_80_cats) +
            f" (these together make up about **{cum_share.iloc[num_for_80-1]*100:.1f}%** of total spending)."
        )

    # ---------- 2) Frequency patterns (how often categories appear) ----------
    cat_counts = df2["Category"].value_counts()
    if not cat_counts.empty:
        freq_cat = cat_counts.index[0]
        freq_count = cat_counts.iloc[0]
        if freq_count >= max(5, 0.3 * len(df2)):
            insights.append(
                f"• You record **{freq_cat}** expenses very frequently "
                f"({int(freq_count)} transactions). Check if any of these can be reduced or optimized."
            )

    # ---------- 3) Merchant / item repetition ----------
    if "Item" in df2.columns:
        item_totals = (
            df2.groupby("Item")["Amount (₹)"]
            .agg(["count", "sum"])
            .sort_values(by="sum", ascending=False)
            .head(5)
            .reset_index()
        )
        if not item_totals.empty:
            top_item = item_totals.iloc[0]
            insights.append(
                f"• **{top_item['Item']}** is one of your top spending sources "
                f"(₹{top_item['sum']:.0f} across {int(top_item['count'])} transactions)."
            )

    # ---------- 4) Daily spending spikes (if Date exists) ----------
    daily_spikes_df = pd.DataFrame()
    if "Date" in df2.columns:
        df_dates = df2.copy()
        df_dates["__date"] = pd.to_datetime(df_dates["Date"], errors="coerce", dayfirst=True)
        df_dates = df_dates.dropna(subset=["__date"])

        if not df_dates.empty:
            by_day = (
                df_dates.groupby("__date")["Amount (₹)"]
                .sum()
                .reset_index()
                .rename(columns={"__date": "Date", "Amount (₹)": "Total (₹)"})
                .sort_values("Date")
            )

            if len(by_day) >= 4:
                median_spend = by_day["Total (₹)"].median()
                std_spend = by_day["Total (₹)"].std(ddof=0)

                if pd.isna(std_spend) or std_spend == 0:
                    threshold = median_spend * 1.5
                else:
                    threshold = median_spend + 2 * std_spend

                daily_spikes_df = by_day[by_day["Total (₹)"] > threshold].copy()

                if not daily_spikes_df.empty:
                    insights.append(
                        "• There are **daily spending spikes** on some dates where your total spending "
                        "is much higher than a normal day. Review those days to see what caused the jump."
                    )

    # ---------- 5) Transaction-level anomaly detection ----------
    # Per-category stats
    cat_stats = (
        df2.groupby("Category")["Amount (₹)"]
        .agg(["mean", "std", "median"])
        .reset_index()
        .rename(
            columns={
                "mean": "cat_mean",
                "std": "cat_std",
                "median": "cat_median",
            }
        )
    )

    df_merged = df2.merge(cat_stats, on="Category", how="left")

    # z-score per category
    df_merged["cat_std_nozero"] = df_merged["cat_std"].replace(0, np.nan)
    df_merged["z_score"] = (df_merged["Amount (₹)"] - df_merged["cat_mean"]) / df_merged["cat_std_nozero"]

    global_median = df_merged["Amount (₹)"].median()

    big_vs_median = df_merged["Amount (₹)"] > (df_merged["cat_median"] * 2)
    high_z = df_merged["z_score"].abs() >= 2
    very_large = df_merged["Amount (₹)"] >= (global_median * 3 if global_median > 0 else 5000)

    mask = big_vs_median | high_z | very_large

    anomalies_df = df_merged[mask].copy().sort_values("Amount (₹)", ascending=False)

    keep_cols = [c for c in ["Date", "Item", "Description", "Amount (₹)", "Category", "Source File", "z_score"] if c in anomalies_df.columns]
    if keep_cols:
        anomalies_df = anomalies_df[keep_cols]

    if not anomalies_df.empty:
        insights.append(
            f"• Detected **{len(anomalies_df)} unusual transactions** that are much larger than your "
            "typical spending in their categories. These may be one-off big purchases, mistakes, or items to review carefully."
        )
    else:
        insights.append(
            "• No strong transaction-level anomalies detected. Your expenses look fairly consistent within this dataset."
        )

    # ---------- Final markdown ----------
    insights_md = "Here are some advanced insights based on your expenses:\n\n" + "\n".join(insights)

    return insights_md, daily_spikes_df, anomalies_df


def compute_spending_trends(df: pd.DataFrame):
    """
    Build dataframes for advanced analytics & trends:
      - daily_totals: date vs total
      - monthly_totals: month (YYYY-MM) vs total
      - cat_month_pivot: month vs category totals
      - weekday_totals: weekday vs total
      - top_merchants: Item vs amount/count

    If no real date column exists, it will create a synthetic date sequence
    (one day per row, ending today) so the dashboard still works for demos.
    """
    if df is None or df.empty:
        return {}

    df2 = df.copy()
    df2["Amount (₹)"] = pd.to_numeric(df2["Amount (₹)"], errors="coerce")
    df2 = df2.dropna(subset=["Amount (₹)"])
    if df2.empty:
        return {}

    # ---- 1) Try to detect a real date column ----
    candidate_cols = [
        "Date",
        "date",
        "Transaction Date",
        "Txn Date",
        "Trans Date",
        "Posted Date",
    ]

    date_col = None
    for c in candidate_cols:
        if c in df2.columns:
            date_col = c
            break

    if date_col is not None:
        df2["__date"] = pd.to_datetime(df2[date_col], errors="coerce", dayfirst=True)
        # keep rows where parsing succeeded
        df_real = df2.dropna(subset=["__date"])
    else:
        df_real = pd.DataFrame()

    # ---- 2) If real dates failed, create synthetic dates ----
    if df_real.empty:
        # use original order, assign a fake date line
        n = len(df2)
        if n == 0:
            return {}

        end_date = pd.Timestamp.today().normalize()
        start_date = end_date - pd.Timedelta(days=n - 1)
        synthetic_dates = pd.date_range(start=start_date, end=end_date, periods=n)

        df2 = df2.reset_index(drop=True)
        df2["__date"] = synthetic_dates
        df_real = df2.copy()

    # from here on, use df_real with a guaranteed __date column
    # ---- 3) Daily totals ----
    daily_totals = (
        df_real.groupby("__date")["Amount (₹)"]
        .sum()
        .reset_index()
        .rename(columns={"__date": "Date", "Amount (₹)": "Total (₹)"})
        .sort_values("Date")
    )

    # ---- 4) Monthly totals ----
    df_real["Month"] = df_real["__date"].dt.to_period("M").astype(str)
    monthly_totals = (
        df_real.groupby("Month")["Amount (₹)"]
        .sum()
        .reset_index()
        .rename(columns={"Amount (₹)": "Total (₹)"})
        .sort_values("Month")
    )

    # ---- 5) Category totals by month (pivot) ----
    if "Category" in df_real.columns:
        cat_month = (
            df_real.groupby(["Month", "Category"])["Amount (₹)"]
            .sum()
            .reset_index()
        )
        cat_month_pivot = cat_month.pivot(
            index="Month", columns="Category", values="Amount (₹)"
        ).fillna(0)
    else:
        cat_month_pivot = pd.DataFrame()

    # ---- 6) Weekday patterns ----
    df_real["Weekday"] = df_real["__date"].dt.day_name()
    weekday_totals = (
        df_real.groupby("Weekday")["Amount (₹)"]
        .sum()
        .reset_index()
        .rename(columns={"Amount (₹)": "Total (₹)"})
    )

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_totals["Weekday"] = pd.Categorical(
        weekday_totals["Weekday"], categories=weekday_order, ordered=True
    )
    weekday_totals = weekday_totals.sort_values("Weekday")

    # ---- 7) Top merchants/items ----
    if "Item" in df_real.columns:
        top_merchants = (
            df_real.groupby("Item")["Amount (₹)"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "Transactions", "sum": "Total (₹)"})
            .sort_values("Total (₹)", ascending=False)
            .head(20)
        )
    else:
        top_merchants = pd.DataFrame()

    return {
        "daily_totals": daily_totals,
        "monthly_totals": monthly_totals,
        "cat_month_pivot": cat_month_pivot,
        "weekday_totals": weekday_totals,
        "top_merchants": top_merchants,
    }




# ---------- GOAL TRACKING TOOL (Tool 4) ----------

GOAL_DEFAULT_CATEGORIES = [
    "Food",
    "Shopping",
    "Entertainment",
    "Transport",
    "Bills",
    "Rent",
    "Education",
    "Loans & EMIs",
    "Investments",
]


def evaluate_goals(df: pd.DataFrame, goal_limits: dict):
    """
    Compare actual spending vs user-defined goals per category.
    goal_limits = {"Food": 3000, "Shopping": 2000, ...}
    Returns DataFrame with Category, Goal, Spent, Remaining, Status.
    """
    if df.empty or not goal_limits:
        return pd.DataFrame()

    spent = df.groupby("Category")["Amount (₹)"].sum().reset_index()
    spent.rename(columns={"Amount (₹)": "Spent (₹)"}, inplace=True)

    rows = []
    for cat, limit in goal_limits.items():
        try:
            limit_val = float(limit)
        except Exception:
            continue
        if limit_val <= 0:
            continue

        cat_spent = float(spent.loc[spent["Category"] == cat, "Spent (₹)"].sum())
        remaining = limit_val - cat_spent

        if remaining < 0:
            status = "Exceeded"
        elif remaining <= 0.1 * limit_val:
            status = "Near Goal"
        else:
            status = "Under Goal"

        rows.append(
            {
                "Category": cat,
                "Goal (₹)": limit_val,
                "Spent (₹)": cat_spent,
                "Remaining (₹)": remaining,
                "Status": status,
            }
        )

    return pd.DataFrame(rows)




def compute_savings_goal_progress(
    df: pd.DataFrame,
    monthly_income: float,
    target_saving_pct: float,
    goals_df: pd.DataFrame | None,
):
    """
    For each user-defined goal (Goal Name, Target Amount, Target Months, Priority),
    estimate required monthly saving and compare with:

      - Approx actual saving: income - total_spent
      - Planned target savings: income * target_saving_pct

    Returns:
      - goals_progress_df: DataFrame with per-goal details
      - approx_saving: float (monthly saving capacity from data)
      - planned_saving: float (your target saving based on %)
    """
    if (
        goals_df is None
        or goals_df.empty
        or df is None
        or df.empty
        or monthly_income <= 0
    ):
        return pd.DataFrame(), 0.0, 0.0

    df2 = df.copy()
    df2["Amount (₹)"] = pd.to_numeric(df2["Amount (₹)"], errors="coerce")
    df2 = df2.dropna(subset=["Amount (₹)"])
    if df2.empty:
        return pd.DataFrame(), 0.0, 0.0

    total_spent = float(df2["Amount (₹)"].sum())
    approx_saving = max(monthly_income - total_spent, 0.0)
    planned_saving = monthly_income * (target_saving_pct / 100.0)

    rows = []

    for _, row in goals_df.iterrows():
        name = str(row.get("Goal Name", "")).strip()
        if not name:
            continue

        goal_type = str(row.get("Goal Type", "")).strip() or "Other"

        try:
            target_amt = float(row.get("Target Amount (₹)", 0.0))
        except Exception:
            target_amt = 0.0

        try:
            target_months = int(row.get("Target Months", 0))
        except Exception:
            target_months = 0

        try:
            priority = int(row.get("Priority (1-5)", 3))
        except Exception:
            priority = 3

        linked_cat = str(row.get("Linked Category (optional)", "")).strip()

        if target_amt <= 0 or target_months <= 0:
            continue

        required_per_month = target_amt / max(target_months, 1)

        if approx_saving <= 0:
            share_of_capacity = None
        else:
            share_of_capacity = required_per_month / approx_saving

        # Status logic
        if approx_saving <= 0:
            status = "No Saving Capacity (Right Now)"
        else:
            if share_of_capacity <= 0.4:
                status = "Comfortable"
            elif share_of_capacity <= 0.8:
                status = "Manageable"
            elif share_of_capacity <= 1.0:
                status = "Very Tight"
            else:
                status = "Unrealistic with Current Pattern"

        rows.append(
            {
                "Goal Name": name,
                "Goal Type": goal_type,
                "Priority (1-5)": priority,
                "Target Amount (₹)": target_amt,
                "Target Months": target_months,
                "Required Monthly (₹)": required_per_month,
                "Approx Monthly Saving Capacity (₹)": approx_saving,
                "Share of Saving Capacity (%)": round(
                    (share_of_capacity * 100.0) if share_of_capacity is not None else 0.0,
                    1,
                ),
                "Linked Category": linked_cat,
                "Status": status,
            }
        )

    if not rows:
        return pd.DataFrame(), approx_saving, planned_saving

    goals_progress_df = pd.DataFrame(rows).sort_values(
        by=["Priority (1-5)", "Required Monthly (₹)"], ascending=[False, False]
    )

    return goals_progress_df, approx_saving, planned_saving




def build_comprehensive_financial_plan_summary(
    goals_progress_df: pd.DataFrame,
    approx_saving: float,
    planned_saving: float,
):
    """
    Build a human-readable summary for comprehensive financial planning.

    Uses:
      - total monthly requirement for all goals
      - your approximate saving capacity
      - your planned saving target
    """
    if goals_progress_df is None or goals_progress_df.empty:
        return (
            "No valid long-term savings/investment goals defined yet. "
            "Add at least one goal with target amount and months to see a full plan."
        )

    total_required = float(goals_progress_df["Required Monthly (₹)"].sum())

    if approx_saving <= 0:
        saving_msg = (
            "Right now, your data suggests **no free saving capacity** "
            "(spending ≈ or > income)."
        )
    else:
        saving_msg = (
            f"Your approximate saving capacity based on expenses is around "
            f"**₹{approx_saving:,.0f} per month**."
        )

    if planned_saving > 0:
        planned_msg = (
            f"Your target savings (based on {planned_saving / max(1, approx_saving + planned_saving) * 100:.1f}% setting) "
            f"is about **₹{planned_saving:,.0f} per month**."
        )
    else:
        planned_msg = "You have not set a clear savings % target, or it is zero."

    lines = []
    lines.append("### 🧭 Comprehensive Financial Plan – Summary")
    lines.append("")
    lines.append(
        f"- Total required per month to hit **all defined goals on time**: "
        f"**₹{total_required:,.0f}**"
    )
    lines.append(f"- {saving_msg}")
    lines.append(f"- {planned_msg}")
    lines.append("")

    if approx_saving <= 0:
        lines.append(
            "➡️ With the current pattern, goals are **not realistically fundable**. "
            "You may need to reduce expenses or increase income before aggressively pursuing these goals."
        )
    else:
        ratio = total_required / approx_saving
        if ratio <= 0.7:
            lines.append(
                "✅ Your goals look **comfortably achievable** within your current saving capacity. "
                "Just stay consistent with your monthly contributions."
            )
        elif ratio <= 1.0:
            lines.append(
                "🟡 Your goals are **manageable but tight**. Small overspending can easily push you off track. "
                "Prioritize high-priority goals (Priority 4–5) and keep strict control on wants."
            )
        else:
            lines.append(
                "🔴 Your goals, as currently defined, require **more than your saving capacity**. "
                "Consider increasing the time horizons for low-priority goals, lowering some target amounts, "
                "or focusing only on the top 1–2 goals first."
            )

    # Count statuses
    comfy = (goals_progress_df["Status"] == "Comfortable").sum()
    manage = (goals_progress_df["Status"] == "Manageable").sum()
    tight = (goals_progress_df["Status"] == "Very Tight").sum()
    unreal = (goals_progress_df["Status"] == "Unrealistic with Current Pattern").sum()

    lines.append("")
    lines.append("**Goal Status Snapshot:**")
    lines.append(f"- Comfortable: **{comfy}**")
    lines.append(f"- Manageable: **{manage}**")
    lines.append(f"- Very Tight: **{tight}**")
    lines.append(f"- Unrealistic with current pattern: **{unreal}**")

    lines.append(
        "\nTip: Start by fully funding your **Emergency / Priority 5** goals, "
        "then move to Priority 4, and so on."
    )

    return "\n".join(lines)





def extract_text_from_guru_docs(files):
    """Read uploaded PDFs/TXT and return a big combined text string."""
    if not files:
        return ""

    all_text = []

    for f in files:
        name = f.name.lower()
        try:
            if name.endswith(".txt"):
                txt = f.read().decode("utf-8", errors="ignore")
                all_text.append(txt)
            elif name.endswith(".pdf"):
                reader = PdfReader(io.BytesIO(f.getvalue()))
                for page in reader.pages:
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    all_text.append(page_text)
        except Exception as e:
            # don’t crash the app if one file fails
            st.warning(f"Could not read {f.name}: {e}")

    # keep it to a reasonable size to avoid overloading anything later
    big_text = "\n".join(all_text)
    if len(big_text) > 20000:
        big_text = big_text[:20000]
    return big_text




def parse_manual_text_expenses(text: str):
    """
    Parse free-text manual entries.
    Each line should contain some description + a number (amount).
    Example lines:
      'Zomato dinner 450'
      'Petrol bike 900'
      'Electricity bill 1200'
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    expenses = []

    for line in lines:
        # find last number in the line as amount
        matches = re.findall(r'([0-9]+(?:\.[0-9]{1,2})?)', line)
        if not matches:
            continue
        amount = float(matches[-1])

        # description = line without the last number
        desc_part = re.sub(r'([0-9]+(?:\.[0-9]{1,2})?)\s*$', "", line).strip()
        if not desc_part:
            desc_part = "Manual Expense"

        # clean item name
        item_name = re.sub(r'[^a-zA-Z ]', ' ', desc_part).strip().title() or "Manual Expense"

        category = detect_category(desc_part)

        expenses.append({
            "Item": item_name,
            "Amount (₹)": amount,
            "Category": category,
            "Source File": "Manual Entry",
        })

    return expenses




def parse_generic_expense_csv(uploaded_file):
    """
    Parse a generic CSV into our format: Item, Amount (₹), Category.
    Tries to guess which columns are description and amount.
    """
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read generic CSV: {e}")
        return pd.DataFrame()

    cols_lower = {c.lower(): c for c in df_raw.columns}
    colnames_lower = list(cols_lower.keys())

    def find_col(*keywords):
        for key in keywords:
            key = key.lower()
            for cname in colnames_lower:
                if key in cname:
                    return cols_lower[cname]
        return None

    desc_col = find_col("description", "details", "narration", "notes", "remark", "particular")
    amount_col = find_col("amount", "amt", "value", "money", "inr", "rs", "rupees")

    # Fallback for amount: first numeric-looking column
    if amount_col is None:
        for c in df_raw.columns:
            try:
                sample = df_raw[c].dropna().head(5).astype(str)
                ok = 0
                for v in sample:
                    v = v.replace(",", "").strip()
                    float(v)
                    ok += 1
                if ok >= max(1, len(sample) // 2):
                    amount_col = c
                    break
            except Exception:
                continue

    # Fallback for description: any object/text column
    if desc_col is None:
        for c in df_raw.columns:
            if df_raw[c].dtype == "object":
                desc_col = c
                break

    if not desc_col or not amount_col:
        st.error("Could not detect Description/Amount columns in generic CSV.")
        return pd.DataFrame()

    expenses = []

    for _, row in df_raw.iterrows():
        raw_amt = str(row[amount_col])
        try:
            amt = float(raw_amt.replace(",", "").strip())
        except Exception:
            continue
        if amt == 0:
            continue

        desc = str(row[desc_col]) if not pd.isna(row[desc_col]) else "CSV Expense"
        desc_clean = desc.strip()
        item_name = re.sub(r'[^a-zA-Z ]', ' ', desc_clean).strip().title() or "Csv Expense"

        category = detect_category(desc_clean)

        expenses.append({
            "Item": item_name,
            "Amount (₹)": amt,
            "Category": category,
            "Source File": f"CSV: {uploaded_file.name}",
        })

    if not expenses:
        return pd.DataFrame()

    return pd.DataFrame(expenses)







# ---------- Streamlit UI ----------

st.set_page_config(page_title="Expense + Financial Advisor", page_icon="💰", layout="centered")
st.title("💰 Smart Expense Categorizer + Financial Advisor")

st.markdown("### ✍️ Optional: Manual Expense Entry")

manual_text = st.text_area(
    "Type expenses (one per line). Example:\n"
    "Zomato dinner 450\n"
    "Petrol bike 900\n"
    "Electricity bill 1200",
    height=120,
)


st.markdown("### 💸 Step 2: Set your basic budget")

col_income, col_saving = st.columns(2)

with col_income:
    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=0.0,
        step=1000.0,
        value=30000.0,
        help="Rough total income per month. Used to compute budgets."
    )

with col_saving:
    target_saving_pct = st.slider(
        "Target Savings (%)",
        min_value=5,
        max_value=60,
        step=5,
        value=20,
        help="What percentage of income you want to save every month."
    )








st.markdown("### 📚 Step 3: Choose your financial philosophy")

all_gurus = list(GURU_PROFILES.keys())

selected_gurus = st.multiselect(
    "Select one or more financial gurus whose style you prefer:",
    all_gurus,
    default=["Warren Buffett"],
)

# 1️⃣ Upload guru documents
guru_docs = st.file_uploader(
    "Optional: Upload financial books/articles/notes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# 2️⃣ Intelligent multi-book processing
guru_notes_text, guru_analysis = analyze_multiple_guru_docs(guru_docs)






st.markdown("### 🎯 Step 4: Optional Category Goals")

goal_limits = {}

with st.expander("Set monthly limits for specific categories (optional):"):
    st.caption("Only categories with a value > 0 will be tracked as goals.")

    for cat in GOAL_DEFAULT_CATEGORIES:
        val = st.number_input(
            f"{cat} monthly limit (₹)",
            min_value=0.0,
            step=500.0,
            value=0.0,
            key=f"goal_{cat}",
        )
        if val > 0:
            goal_limits[cat] = val



st.markdown("### 🧱 Optional: Long-Term Savings & Investment Goals")

with st.expander("Define savings / investment goals (laptop, emergency fund, etc.)"):
    st.caption(
        "Examples:\n"
        "- Emergency Fund → ₹50,000 → 6 months → High priority\n"
        "- New Phone → ₹20,000 → 4 months → Medium priority\n"
        "- Long-term Investment Corpus → ₹2,00,000 → 24 months → Medium/Low priority\n\n"
        "Fill in Goal Name, Type (Emergency / Purchase / Investment / Other), Target Amount, Target Months, and Priority (1–5)."
    )

    # Initialize goals in session_state with richer structure
    if "savings_goals_df" not in st.session_state:
        st.session_state["savings_goals_df"] = pd.DataFrame(
            [
                {
                    "Goal Name": "Emergency Fund",
                    "Goal Type": "Emergency",
                    "Target Amount (₹)": 50000.0,
                    "Target Months": 6,
                    "Priority (1-5)": 5,
                    "Linked Category (optional)": "Investments",
                }
            ]
        )

    goals_df = st.data_editor(
        st.session_state["savings_goals_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="savings_goals_editor",
    )

    if st.button("💾 Save Savings Goals"):
        st.session_state["savings_goals_df"] = goals_df
        st.success("Goals saved for this session. They will be used in the planning section below.")






DOC_MODES = [
    "Auto-detect",
    "Receipts / Expense Images",
    "Bank Statements",
    "UPI / Payment Screenshots",
]


doc_mode = st.radio(
    "📄 What type of financial document are you uploading?",
    DOC_MODES,
    index=0,
    help="Choose 'Bank Statements' for full statements, or 'Receipts' for single bills."
)


uploaded_files = st.file_uploader(
    "📸 Upload one or MANY receipts / bills / statements (images or PDFs)",
    type=SUPPORTED_EXTENSIONS,
    accept_multiple_files=True,
)



st.markdown("### 🧑‍🤝‍🧑 Optional: Add Splitwise group expenses")

splitwise_csv = st.file_uploader(
    "Upload Splitwise CSV export (group expenses)",
    type=["csv"],
    help="Export your data from Splitwise as CSV and upload it here.",
)



st.markdown("### 📂 Optional: Other Expense CSV (Bank / Excel exports)")

generic_csv = st.file_uploader(
    "Upload a generic expense CSV (e.g., from bank app / Excel)",
    type=["csv"],
    key="generic_csv_uploader",
)




if uploaded_files or splitwise_csv is not None or generic_csv is not None or manual_text.strip():
    if st.button("🔍 Analyze & Extract All Expenses"):
        all_expenses = []

        # 1) Process images / PDFs (existing logic)
        if uploaded_files:
            for idx, uploaded in enumerate(uploaded_files, start=1):
                st.markdown(f"### 📄 File {idx}: **{uploaded.name}**")

                try:
                    text, preview_image = advanced_ocr(uploaded)
                except Exception as e:
                    st.error(f"❌ OCR failed for {uploaded.name}: {e}")
                    continue

                if preview_image is not None:
                    st.image(preview_image, caption=f"Preview of {uploaded.name}", use_column_width=True)

                with st.expander(f"🧾 Show extracted text ({uploaded.name})"):
                    st.text_area(
                        f"OCR Output - {uploaded.name}",
                        value=text,
                        height=150,
                    )

                # Decide which parser to use (your existing doc_mode logic)
                if doc_mode == "Bank Statements":
                    parsed = parse_bank_statement(text)
                    source_type = "Bank Statement"

                elif doc_mode == "Receipts / Expense Images":
                    if looks_like_fee_receipt(text):
                        parsed = parse_receipt_total(text)
                    elif looks_like_itemized_receipt(text):
                        parsed = parse_itemized_receipt(text)
                    else:
                        parsed = parse_expenses(text)
                    source_type = "Receipt"

                elif doc_mode == "UPI / Payment Screenshots":
                    parsed = parse_upi_transactions(text)
                    source_type = "UPI / Payment"

                else:
                    # Auto-detect mode
                    if looks_like_bank_statement(text):
                            parsed = parse_bank_statement(text)
                            source_type = "Bank Statement"
                    elif looks_like_upi_block(text):
                            parsed = parse_upi_transactions(text)
                            source_type = "UPI / Payment"
                    elif looks_like_fee_receipt(text):
                            parsed = parse_receipt_total(text)
                            source_type = "Receipt"
                    elif looks_like_itemized_receipt(text):
                            parsed = parse_itemized_receipt(text)
                            source_type = "Receipt"
                    elif looks_like_payment_message_block(text):
                            parsed = parse_payment_sms_messages(text)
                            source_type = "Payment Messages"
                    else:
                            parsed = parse_expenses(text)
                            source_type = "Receipt / Generic OCR"


                for row in parsed:
                    row["Source File"] = uploaded.name
                    row.setdefault("Source Type", source_type)
                all_expenses.extend(parsed)


        # 2) Process Splitwise CSV (NEW)
        splitwise_df = None
        if splitwise_csv is not None:
            st.markdown("### 🧑‍🤝‍🧑 Splitwise Group Expenses (Your Share)")
            splitwise_df = parse_splitwise_csv(splitwise_csv)
            if splitwise_df is not None and not splitwise_df.empty:
                splitwise_df["Source File"] = f"Splitwise: {splitwise_csv.name}"
                splitwise_df["Source Type"] = "Splitwise CSV"

                st.dataframe(splitwise_df, use_container_width=True)

                # group analysis
                if "Group" in splitwise_df.columns:
                    st.markdown("#### Group-wise totals")
                    group_totals = splitwise_df.groupby("Group")["Amount (₹)"].sum()
                    st.bar_chart(group_totals)

                if "Person" in splitwise_df.columns:
                    st.markdown("#### Person-wise totals")
                    person_totals = splitwise_df.groupby("Person")["Amount (₹)"].sum()
                    st.dataframe(person_totals.rename("Total (₹)"))

                # add to unified list
                all_expenses.extend(splitwise_df.to_dict(orient="records"))
            else:
                st.info("Splitwise CSV did not contain usable expenses.")




        # 3) Manual text expenses (NEW)
        if manual_text.strip():
            st.markdown("### ✍️ Manual Expenses")
            manual_expenses = parse_manual_text_expenses(manual_text)
            if manual_expenses:
                manual_df = pd.DataFrame(manual_expenses)
                st.dataframe(manual_df, use_container_width=True)
                manual_df["Source Type"] = "Manual Text"
                all_expenses.extend(manual_df.to_dict(orient="records"))
                # all_expenses.extend(manual_expenses)
            else:
                st.info("No valid items or amounts found in manual text.")

        # 4) Generic CSV expenses (NEW)
        if generic_csv is not None:
            st.markdown("### 📂 Generic CSV Expenses")
            gen_df = parse_generic_expense_csv(generic_csv)
            if gen_df is not None and not gen_df.empty:
                gen_df["Source Type"] = "Generic CSV"
                st.dataframe(gen_df, use_container_width=True)
                all_expenses.extend(gen_df.to_dict(orient="records"))

            else:
                st.info("Generic CSV did not contain usable expenses.")
                


        
        # 3) Unified analysis (all sources merged)
        if not all_expenses:
            st.warning("No valid expenses found in any source. Try clearer files or check formats.")
        else:
            df = pd.DataFrame(all_expenses)

            # ---------- 🔧 TOOL 1: Unified Expense Extraction + Patterns ----------
            st.subheader("🔧 Tool 1 – Unified Expense Extraction")
            st.markdown("All expenses combined from receipts, PDFs, Splitwise, manual inputs, and CSVs.")

            st.caption(
                "You can edit the **Category** column below. When you save, the app will learn from your changes "
                "and automatically categorize similar transactions in the future."
            )

            # Editable table so user can fix categories
            editable_df = st.data_editor(
                df,
                use_container_width=True,
                key="editable_expense_table",
            )

            # Button to learn from corrections
            if st.button("💾 Learn from Category Corrections"):
                changed_rows = []

                # Compare original df vs edited_df
                for idx in editable_df.index:
                    try:
                        old_cat = str(df.loc[idx, "Category"])
                        new_cat = str(editable_df.loc[idx, "Category"])
                    except Exception:
                        continue

                    if not new_cat.strip():
                        continue

                    if old_cat != new_cat:
                        changed_rows.append((idx, new_cat))

                if not changed_rows:
                    st.info("No category changes detected.")
                else:
                    # Ensure a session map exists
                    if "custom_category_map" not in st.session_state:
                        st.session_state["custom_category_map"] = {}

                    # Start from global map
                    for k, v in CUSTOM_CATEGORY_MAP.items():
                        st.session_state["custom_category_map"][k] = v

                    # Build patterns from corrected rows
                    for idx, new_cat in changed_rows:
                        row = editable_df.loc[idx]

                        # Use Item + optional Description to build a pattern
                        base_text = f"{row.get('Item', '')} {row.get('Description', '')}"
                        norm = re.sub(r"[^a-zA-Z0-9 ]", " ", str(base_text)).strip().lower()
                        tokens = [t for t in norm.split() if len(t) > 2][:3]  # first few meaningful words

                        if not tokens:
                            continue

                        patt = " ".join(tokens)
                        st.session_state["custom_category_map"][patt] = new_cat

                    # Push to global map
                    CUSTOM_CATEGORY_MAP.clear()
                    CUSTOM_CATEGORY_MAP.update(st.session_state["custom_category_map"])

                    # Persist to CSV
                    try:
                        save_custom_category_map()
                        st.success(
                            "Learned from your corrections! Similar future expenses will be auto-categorized "
                            "using these patterns."
                        )
                    except Exception as e:
                        st.warning(f"Learning stored for this session, but file save failed: {e}")

                # Use the edited categories for the rest of the analysis
                df = editable_df.copy()
            else:
                # Even if user didn't click learn, keep any edits for this run
                df = editable_df.copy()

            # Now recompute totals based on (possibly) updated df
            total = df["Amount (₹)"].sum()
            by_cat = df.groupby("Category")["Amount (₹)"].sum().reset_index()

            st.markdown(f"### 💰 Total Expense (All Sources): {format_inr(total, 0)}")

            st.bar_chart(by_cat.set_index("Category"))


                        # ---------- 🔄 Automation Summary: Source-wise ----------
            st.subheader("🔄 Automation Summary – Sources & Channels")

            if "Source Type" in df.columns:
                src_summary = (
                    df.groupby("Source Type")["Amount (₹)"]
                    .sum()
                    .reset_index()
                    .sort_values("Amount (₹)", ascending=False)
                )
                st.markdown("#### Source-wise Total Expenses")
                st.dataframe(src_summary, use_container_width=True)

            # UPI / Payment channel breakdown (if present)
            channel_cols = [c for c in ["Channel", "Type"] if c in df.columns]
            if "Channel" in df.columns:
                st.markdown("#### UPI / Card / Bank Channel Breakdown")
                ch_summary = (
                    df.groupby("Channel")["Amount (₹)"]
                    .sum()
                    .reset_index()
                    .sort_values("Amount (₹)", ascending=False)
                )
                st.dataframe(ch_summary, use_container_width=True)






            # ---------- 📈 Spending Pattern Analysis ----------
            st.subheader("📈 Spending Pattern Analysis")

            # 1. Category-wise percentage share
            st.markdown("### 📌 Category-wise Spending Share (%)")

            by_cat_percent = by_cat.copy()
            total_amount = by_cat_percent["Amount (₹)"].sum()
            if total_amount > 0:
                by_cat_percent["Percentage"] = (by_cat_percent["Amount (₹)"] / total_amount) * 100
            else:
                by_cat_percent["Percentage"] = 0.0
            st.dataframe(by_cat_percent, use_container_width=True)

            # Pie chart for category percentage
            st.markdown("### 🥧 Spending Distribution")

            fig, ax = plt.subplots()
            ax.pie(
                by_cat_percent["Amount (₹)"],
                labels=by_cat_percent["Category"],
                autopct="%1.1f%%"
            )
            ax.axis("equal")
            st.pyplot(fig)

            # 2. Top 5 highest individual expenses
            st.markdown("### 🔝 Top 5 Highest Individual Expenses")
            top5 = df.sort_values("Amount (₹)", ascending=False).head(5)
            st.dataframe(top5, use_container_width=True)

            # 3. Average spending per category
            st.markdown("### 📉 Average Spending per Category")
            avg_cat = df.groupby("Category")["Amount (₹)"].mean().reset_index()
            avg_cat.rename(columns={"Amount (₹)": "Average Amount (₹)"}, inplace=True)
            st.dataframe(avg_cat, use_container_width=True)

            # 4. Needs vs Wants vs Others breakdown
            st.markdown("### 🧩 Needs vs Wants vs Others Breakdown")
            df_bucket = df.copy()
            df_bucket["Bucket"] = df_bucket["Category"].map(CATEGORY_TO_BUCKET)
            bucket_summary = df_bucket.groupby("Bucket")["Amount (₹)"].sum().reset_index()
            st.bar_chart(bucket_summary.set_index("Bucket"))



           
             # ---------- 🧠 TOOL 2: Financial Advice Engine ----------
            st.subheader("🧠 Tool 2 – Financial Advice (Guru-Based)")

            advice = generate_financial_advice(
            df=df,
            selected_gurus=selected_gurus,
            guru_notes_text=guru_notes_text,
            total_spent=total,
            guru_analysis=guru_analysis,
            )

            st.markdown(advice)

                        # ---------- 🧠 TRACK B: Multi-Guru Financial Philosophy Comparison ----------
            st.subheader("🧠 Track B – Multi-Guru Financial Philosophy Comparison")

            st.caption(
                "See how your current spending pattern lines up with each selected guru's core focus areas."
            )

            guru_comp_text, guru_comp_df = build_multi_guru_comparison(
                df=df,
                selected_gurus=selected_gurus,
                guru_analysis=guru_analysis,
            )

            st.markdown(guru_comp_text)

            if guru_comp_df is not None and not guru_comp_df.empty:
                st.markdown("#### Guru Comparison Table")
                st.dataframe(guru_comp_df, use_container_width=True)





            # ---------- 🧮 TOOL 3: Budget Planner (Income + 50-30-20) ----------
            st.subheader("🧮 Tool 3 – Budget vs Spending")

            bucket_budget, cat_budget = compute_budget_allocation(
                monthly_income=monthly_income,
                target_saving_pct=float(target_saving_pct),
            )

            st.markdown("### 📊 Budget vs Spending (All Sources)")
            budget_compare_df = analyze_spending_vs_budget(df, cat_budget)
            if not budget_compare_df.empty:
                st.dataframe(budget_compare_df, use_container_width=True)
                overspent = budget_compare_df[budget_compare_df["Status"] == "High"]["Category"].tolist()
                near_limit = budget_compare_df[budget_compare_df["Status"] == "Near Limit"]["Category"].tolist()
                if overspent:
                    st.error("⚠️ You are overspending in: " + ", ".join(overspent))
                if near_limit:
                    st.warning("🟡 Close to budget limit in: " + ", ".join(near_limit))

            st.info(
                f"Recommended monthly split → Needs ≈ ₹{bucket_budget['Needs']:.0f}, "
                f"Wants ≈ ₹{bucket_budget['Wants']:.0f}, "
                f"Savings ≈ ₹{bucket_budget['Savings']:.0f}"
            )



                        # ---------- 🔮 Smart Expense Prediction & Future Budget Pressure ----------
            st.subheader("🔮 Smart Expense Prediction & Future Budget Pressure")

            pred_df = predict_future_expenses(
                df=df,
                cat_budget=cat_budget,
                horizon_days=30,
            )

            if pred_df is None or pred_df.empty:
                st.info("Not enough data (categories/dates) to generate predictions yet.")
            else:
                st.markdown(
                    "Projected **next 30 days** spending per category based on your current pattern, "
                    "compared with your monthly budgets."
                )
                st.dataframe(pred_df, use_container_width=True)

                risky = pred_df[pred_df["Status"].isin(["Likely Over Budget", "Close to Limit"])]
                if not risky.empty:
                    txt_lines = []
                    col_pred = "Predicted Next 30 Days (₹)"
                    for _, row in risky.iterrows():
                        txt_lines.append(
                            f"- {row['Category']}: predicted ≈ ₹{row[col_pred]:.0f} vs budget ₹{row['Budget (₹)']:.0f}"
                        )
                    st.error(
                        "⚠️ Categories with **future budget pressure** if you continue at this speed:\n\n"
                        + "\n".join(txt_lines)
                    )
                else:
                    st.success(
                        "✅ Based on your current pattern, your predicted spending for the next month stays within your category budgets."
                    )



            

            # ---------- 🧮 Budgeting Recommendations (Text) ----------
            st.subheader("🧮 Budgeting Recommendations (Based on Your Patterns)")

            budget_recs = generate_budget_recommendations(
                df=df,
                budget_compare_df=budget_compare_df,
                monthly_income=monthly_income,
                target_saving_pct=float(target_saving_pct),
                bucket_budget=bucket_budget,
            )
            st.markdown(budget_recs)



            # ---------- 🎯 TOOL 4: Goal Tracking ----------
            st.subheader("🎯 Tool 4 – Goal Tracking (Per-Category Limits)")

            goal_df = evaluate_goals(df, goal_limits)
            if goal_df is None or goal_df.empty:
                st.info("No goals set yet. Use the 'Optional Category Goals' section above to define monthly limits.")
            else:
                st.dataframe(goal_df, use_container_width=True)

                # Progress bars for each goal
                for _, row in goal_df.iterrows():
                    cat = row["Category"]
                    spent_val = row["Spent (₹)"]
                    goal_val = row["Goal (₹)"]
                    st.caption(f"{cat}: spent ₹{spent_val:.0f} of goal ₹{goal_val:.0f}")
                    if goal_val > 0:
                        progress = min(max(spent_val / goal_val, 0), 1)
                        st.progress(progress)


            # ---------- 🧱 Track B – Comprehensive Financial Planning & Goal Tracking ----------
            st.subheader("🧱 Track B – Comprehensive Financial Planning & Goal Tracking")

            user_goals_df = st.session_state.get("savings_goals_df")

            goals_progress_df, approx_saving, planned_saving = compute_savings_goal_progress(
                df=df,
                monthly_income=monthly_income,
                target_saving_pct=float(target_saving_pct),
                goals_df=user_goals_df,
            )

            if goals_progress_df is None or goals_progress_df.empty:
                st.info(
                    "No valid long-term goals found. Use the 'Long-Term Savings & Investment Goals' section "
                    "above to define your targets."
                )
            else:
                # Overall plan summary
                plan_summary = build_comprehensive_financial_plan_summary(
                    goals_progress_df=goals_progress_df,
                    approx_saving=approx_saving,
                    planned_saving=planned_saving,
                )
                st.markdown(plan_summary)

                st.markdown("#### Per-Goal Details")
                st.dataframe(goals_progress_df, use_container_width=True)






             # ---------- 🚨 TOOL 5: Advanced Pattern Recognition & Anomaly Detection ----------
            st.subheader("🚨 Tool 5 – Advanced Pattern Recognition & Anomaly Detection")

            pattern_text, daily_spikes_df, anomalies_df = analyze_patterns_and_anomalies(df)

            if pattern_text:
                st.markdown(pattern_text)

            # Daily spikes (needs Date column to exist)
            if daily_spikes_df is not None and not daily_spikes_df.empty:
                st.markdown("#### 📆 Unusual Daily Spending Spikes")
                st.caption("These are days where your total spending is much higher than your normal daily average.")
                st.dataframe(daily_spikes_df, use_container_width=True)

                # Simple bar chart of spikes
                try:
                    st.bar_chart(daily_spikes_df.set_index("Date")["Total (₹)"])
                except Exception:
                    # In case Date is still string; it's fine to skip chart
                    pass

            # Transaction anomalies
            if anomalies_df is not None and not anomalies_df.empty:
                st.markdown("#### 🚩 Unusual Individual Transactions")
                st.caption("These transactions are much larger than normal for their category/date.")
                st.dataframe(anomalies_df, use_container_width=True)
            else:
                st.info("No strong anomalies found in individual transactions for this dataset.")


                        # ---------- 📊 Track B – Advanced Analytics Dashboard & Trends ----------
            st.subheader("📊 Track B – Advanced Analytics Dashboard & Trends")

            trends = compute_spending_trends(df)

            if not trends:
                st.info(
                    "Date information is missing or insufficient to build trends. "
                    "Make sure your data has a valid Date column for time-based analytics."
                )
            else:
                daily_totals = trends["daily_totals"]
                monthly_totals = trends["monthly_totals"]
                cat_month_pivot = trends["cat_month_pivot"]
                weekday_totals = trends["weekday_totals"]
                top_merchants = trends["top_merchants"]

                tab_overview, tab_time, tab_categories, tab_merchants = st.tabs(
                    ["Overview", "Time Trends", "Category Trends", "Merchant Insights"]
                )

                # -------- Overview Tab --------
                with tab_overview:
                    st.markdown("#### 🔍 High-Level View")

                    if not daily_totals.empty:
                        st.markdown("**Daily Spending Trend**")
                        st.line_chart(
                            daily_totals.set_index("Date")["Total (₹)"]
                        )

                    if not monthly_totals.empty:
                        st.markdown("**Monthly Spending Totals**")
                        st.bar_chart(
                            monthly_totals.set_index("Month")["Total (₹)"]
                        )

                    if not weekday_totals.empty:
                        st.markdown("**Average Behaviour by Weekday**")
                        st.bar_chart(
                            weekday_totals.set_index("Weekday")["Total (₹)"]
                        )

                # -------- Time Trends Tab --------
                with tab_time:
                    st.markdown("#### ⏱️ Time-Based Trends")

                    if not daily_totals.empty:
                        st.markdown("**Daily Total (Zoomed View)**")
                        st.dataframe(daily_totals, use_container_width=True)

                    if not monthly_totals.empty:
                        st.markdown("**Monthly Totals Table**")
                        st.dataframe(monthly_totals, use_container_width=True)

                # -------- Category Trends Tab --------
                with tab_categories:
                    st.markdown("#### 📈 Category Trends Over Time (by Month)")

                    if cat_month_pivot is None or cat_month_pivot.empty:
                        st.info("Not enough category + date data to build category trends.")
                    else:
                        st.dataframe(cat_month_pivot, use_container_width=True)

                        # Let user pick top N categories to visualize
                        total_by_cat = cat_month_pivot.sum(axis=0).sort_values(ascending=False)
                        top_cats = list(total_by_cat.index[:5])

                        selected_cats = st.multiselect(
                            "Select categories to visualize",
                            options=list(cat_month_pivot.columns),
                            default=top_cats,
                        )

                        if selected_cats:
                            st.line_chart(cat_month_pivot[selected_cats])
                        else:
                            st.info("Select at least one category to see the trend chart.")

                # -------- Merchant Insights Tab --------
                with tab_merchants:
                    st.markdown("#### 🏪 Merchant / Item Insights")

                    if top_merchants is None or top_merchants.empty:
                        st.info("No item-level data available for merchant insights.")
                    else:
                        st.markdown("**Top 20 Merchants / Items by Total Spend**")
                        st.dataframe(top_merchants, use_container_width=True)

                        st.markdown("You can use this table to identify where most of your money goes "
                                    "(e.g., specific apps, stores, or recurring merchants).")





            # ---------- 🏅 TOOL 6: Financial Health Score ----------
            st.subheader("🏅 Tool 6 – Comprehensive Financial Health Score")

            # Use goal_df created earlier (can be empty)
            score_result = compute_financial_health_score(
                df=df,
                monthly_income=monthly_income,
                target_saving_pct=float(target_saving_pct),
                bucket_budget=bucket_budget,
                budget_compare_df=budget_compare_df,
                goal_df=goal_df,
            )

            st.metric(
                "Overall Financial Health (0–100)",
                f"{score_result['total']:.0f}",
                help="Higher is better. 80+ is strong, 60–80 is okay, below 60 needs attention."
            )

            st.markdown(f"**Grade:** {score_result['grade']}")
            st.markdown(score_result["summary"])

            if score_result["components"]:
                st.markdown("#### 📊 Score Breakdown by Area")
                comp_df = pd.DataFrame(score_result["components"])
                st.dataframe(comp_df, use_container_width=True)



                        # ---------- 🔮 Track B – Predictive Modeling & Personalized Recommendations ----------
            st.subheader("🔮 Track B – Predictive Modeling & Personalized Recommendations")

            prediction = build_predictive_financial_model(df)
            goals_df = st.session_state.get("savings_goals_df", None)

            if not prediction:
                st.info("Not enough data to generate predictive modeling.")
            else:
                st.markdown(f"### 📈 Next Month Spending Forecast: **₹{prediction['next_month_spend']:,.0f}**")
                st.markdown(f"**Trend:** {prediction['trend_strength']}")

                # Category drift table
                drift = prediction.get("category_drift", {})
                if drift:
                    drift_df = pd.DataFrame(
                        [{"Category": c, "Expected Change (%)": p} for c, p in drift.items()]
                    ).sort_values("Expected Change (%)", ascending=False)
                    st.markdown("#### 📊 Expected Category Drift Next Month")
                    st.dataframe(drift_df, use_container_width=True)

                # Personalized recommendations
                recs = build_personalized_recommendations(
                    prediction=prediction,
                    df=df,
                    savings_capacity=approx_saving,
                    goals_df=goals_df,
                    selected_gurus=selected_gurus,
                )
                st.markdown("### 🎯 Personalized AI Recommendations")
                st.markdown(recs)

 


           
                    # ---------- 🇮🇳 TOOL 7: Indian-Specific Investment & Tax Advice ----------
            st.subheader("🇮🇳 Tool 7 – Indian Investment & Tax-Saving Advice")

            indian_advice = generate_indian_investment_advice(
                df=df,
                monthly_income=monthly_income,
                target_saving_pct=float(target_saving_pct),
            )

            st.markdown(indian_advice)




            # ---------- 🇮🇳 TOOL 8: Tax-Saving Recommendations (India) ----------
            st.subheader("🇮🇳 Tool 8 – Tax-Saving Recommendations (Educational)")

            tax_advice = generate_indian_tax_saving_recommendations(
                df=df,
                monthly_income=monthly_income,
                target_saving_pct=float(target_saving_pct),
            )

            st.markdown(tax_advice)


                        # ---------- 🌐 TRACK B: Live Financial Data (APIs) ----------
            st.subheader("🌐 Track B – Live Financial Data (API Integrations)")

            st.caption(
                "These tools use external financial APIs. Configure endpoints and keys in "
                "`.streamlit/secrets.toml` under the [api] section. All calls have robust error handling "
                "so the app remains stable even if APIs fail."
            )

            col1, col2 = st.columns(2)

            # --- Forex snapshot ---
            with col1:
                st.markdown("#### 💱 INR Forex Snapshot")
                if st.button("Fetch Forex Rates (INR Base)"):
                    fx_data, fx_err = fetch_forex_inr_snapshot()
                    if fx_err:
                        st.error(fx_err)
                    elif not fx_data:
                        st.info("No forex data returned.")
                    else:
                        fx_df = pd.DataFrame(
                            [{"Currency": k, "Rate vs INR": v} for k, v in fx_data.items()]
                        )
                        st.dataframe(fx_df, use_container_width=True)

            # --- Market index/stock snapshot ---
            with col2:
                st.markdown("#### 📈 Market Index / Stock Snapshot")
                symbol = st.text_input(
                    "Symbol (example: NIFTY50 / TCS / INFY)",
                    value="NIFTY50",
                    key="market_symbol_input",
                )
                if st.button("Fetch Market Snapshot"):
                    idx_data, idx_err = fetch_market_index_snapshot(symbol=symbol)
                    if idx_err:
                        st.error(idx_err)
                    elif not idx_data:
                        st.info("No market data returned.")
                    else:
                        st.json(idx_data)

            # --- Mutual fund NAV lookup ---
            st.markdown("#### 🧺 Mutual Fund NAV Lookup (Demo)")
            fund_code = st.text_input(
                "Mutual Fund Code / ID (demo)", value="FUND123", key="mf_code_input"
            )
            if st.button("Fetch MF NAV"):
                mf_data, mf_err = fetch_mutual_fund_nav(fund_code=fund_code)
                if mf_err:
                    st.error(mf_err)
                elif not mf_data:
                    st.info("No mutual fund data returned.")
                else:
                    st.json(mf_data)

            # --- Financial news ---
            st.markdown("#### 📰 Financial News")
            news_query = st.text_input(
                "News query", value="personal finance India", key="news_query_input"
            )
            if st.button("Fetch Financial News"):
                news_items, news_err = fetch_financial_news(query=news_query)
                if news_err:
                    st.error(news_err)
                elif not news_items:
                    st.info("No news articles returned.")
                else:
                    for art in news_items:
                        title = art.get("title", "No title")
                        url = art.get("url", "")
                        st.markdown(f"- [{title}]({url})")





        # # ---------- 📚 Financial Content Insights (Books/Articles) ----------
        # st.subheader("📚 Financial Content Insights (Uploaded Books/Articles)")

        # if not guru_notes_text:
        #     st.info("No financial books/articles uploaded yet. You can upload PDFs or TXT files above to get content-based insights.")
        # elif not guru_analysis:
        #     st.info("Could not extract enough meaningful text from the uploaded documents.")
        # else:
        #     wc = guru_analysis["word_count"]
        #     pages = guru_analysis["approx_pages"]
        #     themes = guru_analysis["themes"]
        #     top_kw = guru_analysis["top_keywords"]

        #     st.markdown(
        #         f"- Approximate content length: **{wc} words** (~**{pages}** pages)\n"
        #     )

        #     if themes:
        #         st.markdown(
        #             "- Detected main themes: " +
        #             ", ".join(f"**{t}**" for t in themes)
        #         )
        #     else:
        #         st.markdown("- No strong specific themes detected (content may be very generic).")

        #     st.markdown("#### 🔑 Top recurring concepts / keywords")
        #     # show as a simple table
        #     kw_df = pd.DataFrame(
        #         [{"Keyword": w, "Count": c} for (w, c) in top_kw]
        #     )
        #     st.dataframe(kw_df, use_container_width=True)

        #     if guru_analysis and guru_analysis.get("is_useful"):
        #         st.success("✅ This looks like a useful financial book/article. Your financial advice will be based on this content.")
        #     elif guru_notes_text:
        #         st.warning(
        #             "⚠️ Uploaded content doesn’t look like a strong financial book. "
        #             "The advisor will automatically pick a top personal finance book (like Rich Dad Poor Dad / I Will Teach You To Be Rich) "
        #             "based on your selected gurus and use that instead."
        #         )



                    # ---------- 📄 Detailed Financial Report & Insights ----------
            st.subheader("📄 Detailed Financial Report")

            report_text = build_detailed_financial_report(
                df=df,
                monthly_income=monthly_income,
                target_saving_pct=float(target_saving_pct),
                bucket_budget=bucket_budget,
                cat_budget=cat_budget,
                budget_compare_df=budget_compare_df,
                pred_df=pred_df if "pred_df" in locals() else None,
            )

            with st.expander("Preview Full Report (Markdown Text)"):
                st.text_area(
                    "Report Preview",
                    value=report_text,
                    height=300,
                )

            st.download_button(
                "⬇️ Download Report (TXT)",
                data=report_text.encode("utf-8"),
                file_name="financial_report.txt",
                mime="text/plain",
                key="download_financial_report_txt",
            )





        # Download combined report
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
        "⬇️ Download Expense Report (CSV)",
        data=csv,
        file_name="expense_report.csv",
        mime="text/csv",
        key="single_source_expense_report_csv",
        )


        



