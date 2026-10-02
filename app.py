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

import requests
import streamlit as st
from langchain.agents import create_agent
from langchain_groq import ChatGroq

import auth

HEADERS = {"User-Agent": "wanderly-travel-chatbot/1.0"}
MODEL_NAME = "openai/gpt-oss-120b"

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

def search_places(location, tag, radius=3000, limit=5):
    try:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
            headers=HEADERS, timeout=15,
        ).json()
        if not geo:
            return f"Could not find a location matching '{location}'."
        lat, lon = geo[0]["lat"], geo[0]["lon"]

        query = f'[out:json][timeout:25];(node{tag}(around:{radius},{lat},{lon});); out {limit};'
        elements = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query}, headers=HEADERS, timeout=15,
        ).json().get("elements", [])

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


# =========================================================
# DOCUMENT GENERATION (for the download buttons)
# =========================================================
 
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
 
 
def itinerary_to_pdf(itinerary: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, f"Travel Itinerary - {itinerary['location']}")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 8, f"Generated on {itinerary['generated']} | {itinerary['num_days']} day(s)")
    if itinerary.get("hotel"):
        pdf.multi_cell(0, 8, f"Suggested stay: {itinerary['hotel']}")
    for day in itinerary["days"]:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, f"Day {day['day']}")
        pdf.set_font("Helvetica", "", 11)
        for a in day["attractions"]:
            pdf.multi_cell(0, 7, f"- Visit {a}")
        if day["restaurant"]:
            pdf.multi_cell(0, 7, f"- Eat at {day['restaurant']}")
    if itinerary.get("budget_text"):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, "Budget Plan")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, itinerary["budget_text"])
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


# --- Google login gate (native Streamlit auth, needs secrets.toml - see README) ---
if not st.user.is_logged_in:
    st.info("Please log in with Google to use Wanderly.")
    st.button("🔐 Log in with Google", on_click=st.login)
    st.stop()


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
        "- I have $300 for 3 days, 2 travelers — plan my budget"
    )
    if st.button("🔄 Reset conversation"):
        for key in ("agent", "key", "messages", "chat_history"):
            st.session_state.pop(key, None)
        st.rerun()

if not groq_api_key:
    st.info("👈 Enter your free Groq API key in the sidebar to start chatting.")
    st.stop()

# Build the model + agent once per key (same create_agent pattern as the
# reference files: model + tools -> agent)
if "agent" not in st.session_state or st.session_state.get("key") != groq_api_key:
    groq_llm = ChatGroq(model=MODEL_NAME, api_key=groq_api_key)
    st.session_state.agent = create_agent(
        model=groq_llm,
        tools=[find_nearby_attractions, find_nearby_restaurants, plan_daily_budget],
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
                response = st.session_state.agent.invoke({"messages": pending_history})
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

# --- Download section: appears once an itinerary has been built ---
if st.session_state.get("last_itinerary"):
    st.markdown("---")
    st.subheader("📥 Download your itinerary")
    itinerary = st.session_state["last_itinerary"]
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download as DOCX", data=itinerary_to_docx(itinerary),
            file_name="itinerary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with col2:
        st.download_button(
            "Download as PDF", data=itinerary_to_pdf(itinerary),
            file_name="itinerary.pdf", mime="application/pdf",
        )
 
