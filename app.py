# app.py
# Wanderly - Local Travel Guide Chatbot (TRV-03)
# Core Concept: Search-enabled Agent
# Key Features: Nearby attractions, restaurants, budget planning, user accounts
# Stack: LangChain (create_agent) + Groq (LLM) + OpenStreetMap (places search)
#        + SQLite/bcrypt username-password auth (auth.py)
#        + Google Sign-In via Streamlit's native st.login() (Authlib/OIDC)
# Both Groq and OpenStreetMap are FREE - no credit card needed anywhere.


import base64
from pathlib import Path
import io
import json
import time
import contextvars
from datetime import datetime
from urllib.parse import quote

import requests
import streamlit as st
from docx import Document
from fpdf import FPDF
from langchain.agents import create_agent
from langchain_groq import ChatGroq

import auth

HEADERS = {"User-Agent": "wanderly-travel-chatbot/1.0 (+https://github.com/ashnoroy/travel_guide2.0)"}
MODEL_NAME = "openai/gpt-oss-120b"

# Per-turn holder the build_itinerary tool writes into (see the tool below).
_ITINERARY_HOLDER = contextvars.ContextVar("wanderly_itinerary_holder", default=None)

auth.init_db()

# Google sign-in only renders once secrets.toml has an [auth] section -
# see .streamlit/secrets.toml.example for the required keys. Until then
# the app just falls back to username/password, no crash.
GOOGLE_CONFIGURED = "auth" in st.secrets


# =========================================================
# TOOLS
# Plain python functions with a docstring -> create_agent
# auto-wraps these as tools, same pattern used in the
# reference files (temp_tool, search_latest_info, etc.)
# =========================================================

# The free public OpenStreetMap servers are shared and often busy or
# rate-limited (especially from cloud hosts), so every call below retries
# and falls back to mirror servers before giving up.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


@st.cache_data(ttl=3600, show_spinner=False)
def _geocode_raw(location):
    """Turn a place name into [{'lat','lon','display_name'}].

    Tries Nominatim first, then Photon (another free OpenStreetMap-based
    geocoder that tends to accept requests from cloud hosts). Returns an
    empty list only when a service answered and found nothing."""
    errors = []
    answered_empty = False

    for attempt in range(2):
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1},
                headers=HEADERS, timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                if data:
                    return data
                answered_empty = True
                break
            errors.append(f"Nominatim HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"Nominatim {type(e).__name__}")
        time.sleep(1.5)

    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": location, "limit": 1},
            headers=HEADERS, timeout=20,
        )
        if r.status_code == 200:
            features = r.json().get("features", [])
            if features:
                lon, lat = features[0]["geometry"]["coordinates"]
                props = features[0].get("properties", {})
                name = ", ".join(str(p) for p in (props.get("name"), props.get("state"), props.get("country")) if p)
                return [{"lat": str(lat), "lon": str(lon), "display_name": name or location}]
            answered_empty = True
        else:
            errors.append(f"Photon HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"Photon {type(e).__name__}")

    if answered_empty:
        return []
    print(f"[wanderly] geocoding failed for {location!r}: {errors}", flush=True)
    raise RuntimeError(f"location lookup failed ({'; '.join(errors)})")


@st.cache_data(ttl=3600, show_spinner=False)
def _overpass_query(query):
    """Run an Overpass query, trying each mirror twice. Returns the elements list."""
    last = "no response"
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=30)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                last = f"HTTP {r.status_code} from {url}"
            except Exception as e:
                last = f"{url}: {e}"
            time.sleep(1)
    print(f"[wanderly] Overpass failed on every server: {last}", flush=True)
    raise RuntimeError(f"places search failed ({last})")


def search_places(location, tag, radius=3000, limit=5):
    try:
        geo = _geocode_raw(location)
        if not geo:
            return f"Could not find a location matching '{location}'."
        lat, lon = geo[0]["lat"], geo[0]["lon"]

        query = f'[out:json][timeout:25];(node{tag}["name"](around:{radius},{lat},{lon});); out {limit * 2};'
        elements = _overpass_query(query)

        named = [e for e in elements if e.get("tags", {}).get("name")][:limit]
        if not named:
            return f"No matching places found near {location}."

        lines = []
        for e in named:
            name = e["tags"]["name"]
            kind = e["tags"].get("tourism") or e["tags"].get("amenity", "place")
            lines.append(f"- {name} ({kind})")
        return f"Found near {location}:\n" + "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def find_nearby_attractions(location: str) -> str:
    """Find tourist attractions and points of interest near a given
    location, for example 'Jaipur, India'."""
    return search_places(location, tag='["tourism"]')


def find_nearby_restaurants(location: str) -> str:
    """Find restaurants and cafes near a given location, for example
    'Jaipur, India'."""
    return search_places(location, tag='["amenity"~"restaurant|cafe"]')


def plan_daily_budget(total_budget: float, num_days: int, num_travelers: int = 1) -> str:
    """Split a total trip budget into a per-day, per-person spending
    plan with a suggested food/activities/transport/buffer breakdown."""
    if num_days <= 0 or num_travelers <= 0:
        return "num_days and num_travelers must be greater than zero."
    per_day = total_budget / num_days
    per_day_person = per_day / num_travelers
    return (
        f"Total budget: {total_budget} for {num_days} day(s), {num_travelers} traveler(s).\n"
        f"Per day (all travelers): {per_day:.2f}\n"
        f"Per day, per person: {per_day_person:.2f}\n"
        f"Suggested split per person/day -> "
        f"Food: {per_day_person * 0.4:.2f}, "
        f"Activities: {per_day_person * 0.3:.2f}, "
        f"Transport: {per_day_person * 0.15:.2f}, "
        f"Buffer: {per_day_person * 0.15:.2f}"
    )


def _geocode(location):
    """Internal helper (not a tool): location name -> (lat, lon, resolved_name)."""
    geo = _geocode_raw(location)
    if not geo:
        return None
    return float(geo[0]["lat"]), float(geo[0]["lon"]), geo[0].get("display_name", location)


def _overpass_raw(lat, lon, tag, radius=3000, limit=6):
    """Internal helper (not a tool): raw named OSM elements matching `tag` near (lat, lon)."""
    query = f'[out:json][timeout:25];(node{tag}["name"](around:{radius},{lat},{lon});); out {limit * 2};'
    elements = _overpass_query(query)
    return [e for e in elements if e.get("tags", {}).get("name")][:limit]


_FALLBACK_LLM = {}  # filled in once the Groq model is created (see bottom of file)


def _llm_places(location, num_days):
    """Fallback when the OpenStreetMap servers can't be reached: ask the Groq
    model itself for well-known attractions and restaurants. Returns
    (attractions, restaurants) as lists of names, or raises if that fails too."""
    llm = _FALLBACK_LLM.get("llm")
    if llm is None:
        raise RuntimeError("model not ready")
    n_attr = max(num_days * 2, 6)
    n_rest = max(num_days, 3)
    prompt = (
        f"List {n_attr} famous real tourist attractions and {n_rest} well-known real "
        f"restaurants or cafes in or near {location}. Reply with ONLY a JSON object, "
        'no other text, in this exact shape: {"attractions": ["..."], "restaurants": ["..."]}'
    )
    text = llm.invoke(prompt).content
    if isinstance(text, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    attractions = [str(a).strip() for a in data.get("attractions", []) if str(a).strip()]
    restaurants = [str(r).strip() for r in data.get("restaurants", []) if str(r).strip()]
    if not attractions:
        raise RuntimeError("model returned no attractions")
    return attractions, restaurants


def build_itinerary(location: str, num_days: int, total_budget: float = None, num_travelers: int = 1) -> str:
    """Build a day-wise travel itinerary for a location, combining nearby attractions
    and a restaurant suggestion per day. Use this whenever the user asks for an
    itinerary, a trip plan, or a document/PDF/DOCX they can download - this tool
    prepares the file for the real download buttons that appear below the chat.
    total_budget is optional; include it if the user mentioned one."""
    try:
        used_fallback = False
        try:
            geo = _geocode(location)
            if not geo:
                return f"Could not find a location matching '{location}'."
            lat, lon, resolved = geo

            attractions = _overpass_raw(lat, lon, '["tourism"]', limit=max(num_days * 2, 6))
            restaurants = _overpass_raw(lat, lon, '["amenity"~"restaurant|cafe"]', limit=max(num_days, 3))
            attractions = [a["tags"]["name"] for a in attractions]
            restaurants = [r["tags"]["name"] for r in restaurants]
            if not attractions:
                raise RuntimeError("no attractions in map data")
        except Exception as map_err:
            # Map servers unreachable/blocked/empty -> don't fail, use the model's
            # own knowledge of well-known places instead.
            print(f"[wanderly] map data failed ({map_err}); using model fallback", flush=True)
            try:
                attractions, restaurants = _llm_places(location, num_days)
            except Exception as llm_err:
                return f"Could not build the itinerary: map data failed ({map_err}); fallback failed ({llm_err})"
            resolved = location
            used_fallback = True

        days = []
        for d in range(num_days):
            day_attractions = attractions[d * 2:(d * 2) + 2] or attractions[:2]
            day_restaurant = restaurants[d % len(restaurants)] if restaurants else None
            days.append({
                "day": d + 1,
                "attractions": day_attractions,
                "restaurant": day_restaurant,
            })

        budget_text = plan_daily_budget(total_budget, num_days, num_travelers) if total_budget else None

        itinerary = {
            "location": resolved,
            "num_days": num_days,
            "hotel": None,
            "days": days,
            "budget_text": budget_text,
            "generated": datetime.now().strftime("%d %b %Y"),
            "source": "AI suggestions (live map data was unavailable)" if used_fallback else "OpenStreetMap",
        }
        # The agent may run tools on a worker thread, where Streamlit's
        # session_state isn't reliably available. So hand the itinerary back
        # through a holder the main script created for this turn, and keep
        # the session_state write as a fallback.
        holder = _ITINERARY_HOLDER.get(None)
        if holder is not None:
            holder["itinerary"] = itinerary
        try:
            st.session_state["last_itinerary"] = itinerary
        except Exception:
            pass

        lines = [f"Itinerary for {resolved} ({num_days} day(s)) is ready:"]
        for day in days:
            lines.append(f"\nDay {day['day']}:")
            lines += [f"  - Visit {a}" for a in day["attractions"]]
            if day["restaurant"]:
                lines.append(f"  - Eat at {day['restaurant']}")
        if budget_text:
            lines.append(f"\n{budget_text}")
        if used_fallback:
            lines.append(
                "\nNote: live map data was unavailable, so these are well-known places "
                "suggested by the AI - tell the user to double-check opening hours."
            )
        lines.append(
            "\n(The DOCX and PDF download buttons for this itinerary have appeared "
            "below the chat - do not try to generate a download link yourself.)"
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Could not build the itinerary: {e}"


# =========================================================
# DOCUMENT GENERATION (for the download buttons)
# =========================================================
 
def check_map_services():
    """Ping each free map service once and report (name, ok, detail). Used by
    the sidebar's 'Test map connection' button to show the real cause when
    place searches fail."""
    results = []
    probes = [
        ("Nominatim (geocoding)", "get", "https://nominatim.openstreetmap.org/search",
         {"params": {"q": "Jaipur", "format": "json", "limit": 1}}),
        ("Photon (geocoding backup)", "get", "https://photon.komoot.io/api/",
         {"params": {"q": "Jaipur", "limit": 1}}),
    ] + [
        (f"Overpass ({u.split('/')[2]})", "post", u,
         {"data": {"data": '[out:json][timeout:10];node["tourism"="attraction"](around:500,26.9124,75.7873);out 1;'}})
        for u in OVERPASS_URLS
    ]
    for name, method, url, kwargs in probes:
        try:
            r = getattr(requests, method)(url, headers=HEADERS, timeout=20, **kwargs)
            results.append((name, r.status_code == 200, f"HTTP {r.status_code}"))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
    return results


def itinerary_to_docx(itinerary: dict) -> bytes:
    doc = Document()
    doc.add_heading(f"Travel Itinerary \u2013 {itinerary['location']}", level=1)
    doc.add_paragraph(f"Generated on {itinerary['generated']} | {itinerary['num_days']} day(s)")
    if itinerary.get("hotel"):
        doc.add_paragraph(f"Suggested stay: {itinerary['hotel']}")
    for day in itinerary["days"]:
        doc.add_heading(f"Day {day['day']}", level=2)
        for a in day["attractions"]:
            doc.add_paragraph(f"Visit: {a}", style="List Bullet")
        if day["restaurant"]:
            doc.add_paragraph(f"Eat at: {day['restaurant']}", style="List Bullet")
    if itinerary.get("budget_text"):
        doc.add_heading("Budget Plan", level=2)
        doc.add_paragraph(itinerary["budget_text"])
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
 
 
def _mc(pdf, h, text):
    """multi_cell that always restarts at the left margin. Newer fpdf2 leaves
    the cursor at the right edge after multi_cell, which made the next full-width
    cell fail with 'Not enough horizontal space to render a single character'."""
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, h, text, new_x="LMARGIN", new_y="NEXT")


def itinerary_to_pdf(itinerary: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _mc(pdf, 10, f"Travel Itinerary - {itinerary['location']}")
    pdf.set_font("Helvetica", "", 11)
    _mc(pdf, 8, f"Generated on {itinerary['generated']} | {itinerary['num_days']} day(s)")
    if itinerary.get("hotel"):
        _mc(pdf, 8, f"Suggested stay: {itinerary['hotel']}")
    for day in itinerary["days"]:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        _mc(pdf, 8, f"Day {day['day']}")
        pdf.set_font("Helvetica", "", 11)
        for a in day["attractions"]:
            _mc(pdf, 7, f"- Visit {a}")
        if day["restaurant"]:
            _mc(pdf, 7, f"- Eat at {day['restaurant']}")
    if itinerary.get("budget_text"):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        _mc(pdf, 8, "Budget Plan")
        pdf.set_font("Helvetica", "", 11)
        _mc(pdf, 7, itinerary["budget_text"])
    return bytes(pdf.output())



def _clean_markdown(text: str) -> list:
    """Chat markdown -> list of (kind, text) lines: kind is 'h' (heading),
    'b' (bullet) or 'p' (paragraph). Strips **bold**, #, tables and emojis."""
    import re
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r"[-*_|:\s]{3,}", line):
            continue
        kind = "p"
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            kind, line = "h", m.group(1)
        elif re.match(r"^[-*\u2022]\s+", line):
            kind, line = "b", re.sub(r"^[-*\u2022]\s+", "", line)
        if line.startswith("|"):
            line = "  ".join(c.strip() for c in line.strip("|").split("|"))
        line = re.sub(r"[*_`]{1,3}", "", line)
        out.append((kind, line.strip()))
    return out


def reply_to_docx(text: str) -> bytes:
    doc = Document()
    doc.add_heading("Wanderly - Travel Guide", level=1)
    doc.add_paragraph(f"Generated on {datetime.now().strftime('%d %b %Y')}")
    for kind, line in _clean_markdown(text):
        if kind == "h":
            doc.add_heading(line, level=2)
        elif kind == "b":
            doc.add_paragraph(line, style="List Bullet")
        else:
            doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def reply_to_pdf(text: str) -> bytes:
    # The built-in PDF fonts only support Latin-1, so swap common symbols
    # and drop emojis/other characters they can't draw.
    swaps = {"\u20b9": "Rs ", "\u2248": "~", "\u2192": "->", "\u2013": "-",
             "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"',
             "\u201d": '"', "\u2022": "-", "\u2026": "..."}

    def latin(t):
        for k, v in swaps.items():
            t = t.replace(k, v)
        return t.encode("latin-1", "ignore").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _mc(pdf, 10, "Wanderly - Travel Guide")
    pdf.set_font("Helvetica", "", 10)
    _mc(pdf, 7, f"Generated on {datetime.now().strftime('%d %b %Y')}")
    pdf.ln(2)
    for kind, line in _clean_markdown(text):
        line = latin(line).strip()
        if not line:
            continue
        if kind == "h":
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            _mc(pdf, 8, line)
        elif kind == "b":
            pdf.set_font("Helvetica", "", 11)
            _mc(pdf, 7, "- " + line)
        else:
            pdf.set_font("Helvetica", "", 11)
            _mc(pdf, 7, line)
    return bytes(pdf.output())



# =========================================================
# STREAMLIT FRONTEND
# =========================================================

st.set_page_config(page_title="Wanderly - Local Travel Guide", page_icon="🧭")


def set_background(image_path: str) -> None:

    path = Path(image_path)
    if not path.exists():
        return  # fail quietly - app still works without a background
    encoded = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lstrip(".")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(10, 20, 20, 0.72), rgba(10, 20, 20, 0.72)),
                url("data:image/{ext};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(10, 20, 20, 0.85);
        }}
        [data-testid="stChatMessage"] {{
            background-color: rgba(20, 30, 30, 0.75);
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background(str(Path(__file__).parent / "assets" / "background.jpg"))

st.title("🧭 Wanderly — Local Travel Guide Chatbot")
st.caption("Nearby attractions & restaurants · Budget planning")


# =========================================================
# AUTH GATE
# Two independent ways in:
#   1. Google Sign-In via Streamlit's native st.login() (needs secrets.toml)
#   2. Username/password via auth.py (SQLite + bcrypt)
# Either one sets st.session_state.auth_user, which is all the rest of the
# app cares about.
# =========================================================

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# Pick up a Google session automatically once Streamlit's own auth cookie
# confirms the user is logged in.
if GOOGLE_CONFIGURED and st.user.is_logged_in and st.session_state.auth_user is None:
    st.session_state.auth_user = st.user.email or st.user.name

if st.session_state.auth_user is None:
    st.subheader("Welcome — please log in or create an account to continue")

    if GOOGLE_CONFIGURED:
        if st.button("🔵 Continue with Google", use_container_width=True):
            st.login("google")
        st.divider()
        st.caption("Or use a Wanderly account:")
    else:
        st.caption(
            "Google Sign-In isn't configured yet on this deployment "
            "(see .streamlit/secrets.toml.example) — use a Wanderly account below."
        )

    login_tab, register_tab = st.tabs(["🔐 Log in", "📝 Register"])

    with login_tab:
        with st.form("login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            success, result = auth.verify_login(identifier, password)
            if success:
                st.session_state.auth_user = result  # result is the username
                st.rerun()
            else:
                st.error(result)

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            register_submitted = st.form_submit_button("Create account")

        if register_submitted:
            success, message = auth.register_user(
                new_username, new_email, new_password, confirm_password
            )
            if success:
                st.success(message + " Switch to the Log in tab to sign in.")
            else:
                st.error(message)

    st.stop()  # don't render anything below until logged in


# =========================================================
# MAIN APP (only reached once auth_user is set)
# =========================================================

with st.sidebar:
    st.markdown(f"👋 Logged in as **{st.session_state.auth_user}**")
    if st.button("➾ Log out"):
        signed_in_with_google = GOOGLE_CONFIGURED and st.user.is_logged_in
        for key in ("auth_user", "agent", "key", "messages", "chat_history"):
            st.session_state.pop(key, None)
        if signed_in_with_google:
            st.logout()  # clears Streamlit's identity cookie and reruns
        else:
            st.rerun()

    st.markdown("---")
    st.header("🔑 Groq API Key")
    st.caption("Free, no credit card. Get one at console.groq.com/keys")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")
    st.markdown(
        "**Try asking:**\n"
        "- Find attractions near Jaipur, India\n"
        "- Suggest restaurants near Amer Fort\n"
        "- I have $300 for 3 days, 2 travelers — plan my budget\n"
        "- Plan a 3-day Jaipur itinerary I can download"
    )
    if st.button("🛠 Test map connection"):
        with st.spinner("Checking the free map services..."):
            for name, ok, detail in check_map_services():
                (st.success if ok else st.error)(f"{name}: {detail}")
    if st.button("🔄 Reset conversation"):
        for key in ("agent", "key", "messages", "chat_history"):
            st.session_state.pop(key, None)
        st.rerun()

if not groq_api_key:
    st.info("👈 Enter your free Groq API key in the sidebar to start chatting.")
    st.stop()

SYSTEM_PROMPT = (
    "You are Wanderly, a friendly local travel guide chatbot. Use your tools to find "
    "real attractions and restaurants, plan budgets, and build itineraries - never "
    "invent place names, prices, or dates; only report what a tool actually returns.\n\n"
    "If the user asks for an itinerary, a trip plan, or anything they can download "
    "as a document/PDF/DOCX, you MUST call the build_itinerary tool rather than "
    "answering directly. Real 'Download as DOCX' and 'Download as PDF' buttons "
    "appear below the chat automatically once that tool runs - never write out an "
    "HTML <a href>, a data: URI, or any raw markup yourself to fake a download link; "
    "you cannot create real files by typing text, so just tell the user in plain "
    "words that their file is ready below the chat.\n\n"
    "Reply in the same language and style the user writes in: if they write in "
    "Hinglish (mixed Hindi-English in Roman script), reply in Hinglish too; if they "
    "write in plain English, reply in English; if they write in Hindi, reply in Hindi. "
    "Keep answers concise and use bullet points for lists of places."
)

# Build the model + agent once per key (same create_agent pattern as the
# reference files: model + tools -> agent)
if "agent" not in st.session_state or st.session_state.get("key") != groq_api_key:
    groq_llm = ChatGroq(model=MODEL_NAME, api_key=groq_api_key)
    _FALLBACK_LLM["llm"] = groq_llm
    st.session_state.agent = create_agent(
        model=groq_llm,
        tools=[find_nearby_attractions, find_nearby_restaurants, plan_daily_budget, build_itinerary],
        system_prompt=SYSTEM_PROMPT,
    )
    st.session_state.key = groq_api_key

if "messages" not in st.session_state:
    st.session_state.messages = []       # for rendering chat bubbles
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # for the agent's "messages" input

# Render past messages
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Ask about attractions, restaurants, or your travel budget...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Only append to the agent's history AFTER a successful reply, so a
    # failed turn doesn't leave a dangling user message that confuses the
    # next call.
    pending_history = st.session_state.chat_history + [{"role": "user", "content": user_input}]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                holder = {}
                _ITINERARY_HOLDER.set(holder)
                response = st.session_state.agent.invoke({"messages": pending_history})
                if holder.get("itinerary"):
                    st.session_state["last_itinerary"] = holder["itinerary"]
                content = response["messages"][-1].content
                # Groq/gpt-oss returns plain string content. Some other
                # providers (e.g. Gemini) return a list of content blocks
                # instead - handle both safely.
                if isinstance(content, list):
                    reply = content[-1].get("text", str(content[-1])) if content else ""
                else:
                    reply = content
                if not reply:
                    reply = "I didn't get a usable response - try rephrasing your question."
                st.session_state.chat_history = pending_history
            except Exception as e:
                reply = (
                    f"Something went wrong calling the model: {e}\n\n"
                    "This is usually a temporary model/tool-calling hiccup - try "
                    "rephrasing your question, or click 'Reset conversation' in the sidebar."
                )
                # Don't save the failed turn into chat_history - keep it clean for retry.
        st.markdown(reply)

    st.session_state.messages.append(("assistant", reply))

# --- Download section ---
# 1) Structured itinerary (when the build_itinerary tool ran)
# 2) Always: the latest assistant answer as DOCX/PDF, so a download is
#    available even if the model answered without calling the tool.
_last_reply = next((t for r, t in reversed(st.session_state.messages) if r == "assistant"), None)
_last_ok = _last_reply and not _last_reply.startswith("Something went wrong")

if st.session_state.get("last_itinerary") or _last_ok:
    st.markdown("---")
    st.subheader("\U0001F4E5 Downloads")

if st.session_state.get("last_itinerary"):
    itinerary = st.session_state["last_itinerary"]
    st.caption("Itinerary file")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download itinerary as DOCX", data=itinerary_to_docx(itinerary),
            file_name="itinerary.docx", key="dl_it_docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with col2:
        st.download_button(
            "Download itinerary as PDF", data=itinerary_to_pdf(itinerary),
            file_name="itinerary.pdf", mime="application/pdf", key="dl_it_pdf",
        )

if _last_ok:
    st.caption("Latest answer from Wanderly")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download last answer as DOCX", data=reply_to_docx(_last_reply),
            file_name="wanderly_answer.docx", key="dl_reply_docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with c2:
        st.download_button(
            "Download last answer as PDF", data=reply_to_pdf(_last_reply),
            file_name="wanderly_answer.pdf", mime="application/pdf", key="dl_reply_pdf",
        )
