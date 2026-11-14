import html
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# Import orchestrator components from main.py
from main import app, store, MODEL_NAME, initial_state

BASE_DIR = Path(__file__).parent

# ── Page Config (Responsive for Mobile & Desktop) ─────────────────────────────
st.set_page_config(
    page_title="AI Travel Booking System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="auto",  # Laptop pe open, Mobile pe clean hamburger icon
)

# ── Load CSS with Responsive Built-in Fallback ────────────────────────────────
css_file = BASE_DIR / "styles.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    # Responsive fallback styling in case styles.css is missing
    st.markdown("""
    <style>
        .sidebar-title { font-size: 1.05rem; font-weight: 700; margin: 1rem 0 0.4rem 0; color: #1E293B; }
        .sidebar-chip { background: #F1F5F9; border-radius: 8px; padding: 6px 12px; margin-bottom: 6px; font-size: 0.82rem; color: #334155; word-break: break-word; }
        .hero-wrapper { position: relative; border-radius: 14px; overflow: hidden; margin-bottom: 1.5rem; background: #0F172A; min-height: 200px; display: flex; align-items: center; justify-content: center; }
        .hero-bg { width: 100%; height: 100%; object-fit: cover; opacity: 0.45; position: absolute; inset: 0; }
        .hero-content { position: relative; z-index: 2; padding: 1.5rem 1rem; display: flex; flex-direction: column; align-items: center; text-align: center; width: 100%; }
        .hero-badge { display: inline-block; background: rgba(255,255,255,0.2); backdrop-filter: blur(8px); padding: 4px 10px; border-radius: 20px; font-size: 0.72rem; color: #fff; margin-bottom: 0.5rem; }
        .hero-title { font-size: clamp(1.4rem, 4vw, 2rem); font-weight: 800; color: #fff; margin: 0; }
        .hero-sub { color: #E2E8F0; font-size: clamp(0.8rem, 2vw, 0.92rem); margin-top: 0.5rem; max-width: 650px; }
        .sec-head { font-size: clamp(1.1rem, 2.5vw, 1.3rem); font-weight: 700; margin: 1.5rem 0 1rem 0; color: #0F172A; }
        .input-label { font-size: clamp(0.95rem, 2vw, 1.1rem); font-weight: 700; margin-bottom: 0.5rem; }
        .metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.75rem; margin: 1.2rem 0; }
        .metric-box { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px; text-align: center; }
        .metric-val { font-size: 1.4rem; font-weight: 700; color: #2563EB; }
        .metric-lbl { font-size: 0.75rem; color: #64748B; text-transform: uppercase; }
        .save-bar { background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; padding: 8px 14px; border-radius: 8px; font-size: 0.85rem; word-break: break-all; }
    </style>
    """, unsafe_allow_html=True)

# ── Session State Management ──────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def set_query(text: str):
    st.session_state.query = text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-title'>✈️ Your Next Adventure Starts Here! ✨</div>", unsafe_allow_html=True)
    st.markdown("---")

    user_id = st.text_input(
        "👤 User ID",
        value="rohit_user",
        help="Your unique ID helps us remember your preferences across all your trips. 🧠",
    ).strip() or "anonymous"

    st.markdown("<div class='sidebar-title'>🧠 Your Travel Preferences</div>", unsafe_allow_html=True)
    remembered = []
    if store is not None:
        try:
            remembered = store.search(("users", user_id, "prefs"), limit=25)
        except Exception:
            remembered = []

    if remembered:
        for m in remembered:
            val = getattr(m, "value", {})
            pref_text = val.get("text") if isinstance(val, dict) else str(val)
            if pref_text:
                safe_text = html.escape(str(pref_text))
                st.markdown(f"<div class='sidebar-chip'>✨ {safe_text}</div>", unsafe_allow_html=True)
    else:
        st.caption("No preferences saved yet. Share your interests while planning your trip!")

    st.markdown("<div class='sidebar-title'>Powered by</div>", unsafe_allow_html=True)
    for tech in ["🔗 LangGraph", f"🧠 Groq · {MODEL_NAME}", "🐘 PostgreSQL / Store", "🔍 Tavily Search"]:
        st.markdown(f"<div class='sidebar-chip'>{tech}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>Agent Pipeline</div>", unsafe_allow_html=True)
    for step in ["① Parse", "② Flight + Hotel (parallel)", "③ Budget (Python math)", "④ Planner", "⑤ Memory"]:
        st.markdown(f"<div class='sidebar-chip'>{step}</div>", unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <img class="hero-bg"
         src="https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1400&q=80"
         alt="airplane above clouds"/>
    <div class="hero-content">
        <div class="hero-badge">✦ Multi-Agent AI System</div>
        <div class="hero-title">✈️ AI Travel Booking System</div>
        <div class="hero-sub">Our AI agents find flights and hotels, remember your preferences, and create a personalized travel experience just for you. ✈️🌍</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Popular Destinations ──────────────────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
    <span>🌍 Where Will Your Next Adventure Take You? ✈️</span>
</div>
""", unsafe_allow_html=True)

DESTINATIONS = [
    {"country": "🇯🇵 Japan", "places": "Tokyo • Kyoto • Osaka",
     "image": "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=700&q=85",
     "url": "https://www.japan.travel/en/"},
    {"country": "🇺🇸 United States", "places": "New York • Los Angeles",
     "image": "https://images.unsplash.com/photo-1534430480872-3498386e7856?w=700&q=85",
     "url": "https://www.visittheusa.com/"},
    {"country": "🇫🇷 France", "places": "Paris • Nice • Lyon",
     "image": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=700&q=85",
     "url": "https://www.france.fr/en/"},
    {"country": "🇦🇪 UAE", "places": "Dubai • Abu Dhabi",
     "image": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=700&q=85",
     "url": "https://www.visitdubai.com/"},
    {"country": "🇮🇩 Indonesia", "places": "Bali • Jakarta",
     "image": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=700&q=85",
     "url": "https://www.indonesia.travel/"},
    {"country": "🇨🇭 Switzerland", "places": "Zurich • Interlaken",
     "image": "https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=700&q=85",
     "url": "https://www.myswitzerland.com/"},
]

for start in range(0, len(DESTINATIONS), 3):
    cols = st.columns(3)
    for col, d in zip(cols, DESTINATIONS[start:start + 3]):
        with col:
            with st.container(border=True):
                st.image(d["image"], use_container_width=True)
                st.markdown(f"### {d['country']}")
                st.caption(d["places"])
                st.link_button("🌍 Explore Destination", url=d["url"], use_container_width=True)

# ── User Input & Quick Prompt Buttons ─────────────────────────────────────────
st.markdown("<div class='input-label'>🗺️ ✈️ Plan Your Dream Trip</div>", unsafe_allow_html=True)

QUICK = {
    "🇯🇵 🌸 Japan Under ₹2L": "7 day trip from Delhi to Tokyo and Kyoto, Japan, for 1 traveller, budget ₹200000",
    "🇫🇷 🗼 Paris Romance": "5 day romantic trip from Mumbai to Paris, France for 2 travellers, mid-range hotel",
    "🇦🇪 🏙️ Dubai Weekend": "3 day luxury weekend from Delhi to Dubai, UAE for 2 travellers, 5-star hotel",
    "🇮🇩 🌴 Bali Bliss": "10 day trip from Bangalore to Bali, Indonesia for 2 travellers, beach and culture",
}

cols = st.columns(len(QUICK))
for qc, (label, full_query) in zip(cols, QUICK.items()):
    qc.button(label, key=f"q_{label}", on_click=set_query, args=(full_query,), use_container_width=True)

user_query = st.text_area(
    "trip",
    key="query",
    height=100,
    label_visibility="collapsed",
    placeholder="✨ Where do you want to go? Share your origin, destination, budget, dates, and preferences. Let AI plan your getaway!",
)

generate = st.button("🚀 Build My Dream Trip", use_container_width=True, type="primary")

# ── Node Display Metadata ─────────────────────────────────────────────────────
AGENT_META = {
    "parse":        ("🧩", "Parse Agent"),
    "flight_agent": ("✈️", "Flight Agent"),
    "hotel_agent":  ("🏨", "Hotel Agent"),
    "budget":       ("🧮", "Budget Agent"),
    "planner":      ("🗓️", "Planner Agent"),
    "memory":       ("🧠", "Memory Agent"),
}

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if generate:
    clean_query = user_query.strip()
    if not clean_query:
        st.warning("Please describe where you would love to travel! ✈️")
        st.stop()

    config = {
        "configurable": {
            "thread_id": f"{user_id}-{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
        }
    }

    collected = {
        "query": clean_query,
        "user_id": user_id,
        "trip": {},
        "flight_results": "",
        "hotel_results": "",
        "budget_md": "",
        "itinerary": "",
        "llm_calls": 0,
        "agents_run": 0,
        "errors": [],
    }

    st.markdown("---")
    st.markdown("<div class='sec-head'><span>🤖 Agent Pipeline Running</span></div>", unsafe_allow_html=True)

    try:
        with st.spinner("AI Agents researching flights, hotels, and building your plan..."):
            for chunk in app.stream(
                initial_state(clean_query),
                config=config,
                stream_mode="updates",
            ):
                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue

                    icon, label = AGENT_META.get(node_name, ("🔧", node_name))
                    collected["agents_run"] += 1
                    collected["llm_calls"] += update.get("llm_calls", 0)
                    collected["errors"] += update.get("errors", [])

                    with st.status(f"{icon}  {label}", state="complete", expanded=(node_name == "planner")):
                        if node_name == "parse":
                            collected["trip"] = update.get("trip", {})
                            st.json(collected["trip"])
                        elif node_name == "flight_agent":
                            collected["flight_results"] = update.get("flight_results", "")
                            st.markdown(collected["flight_results"] or "_No flight data found._")
                        elif node_name == "hotel_agent":
                            collected["hotel_results"] = update.get("hotel_results", "")
                            st.markdown(collected["hotel_results"] or "_No hotel data found._")
                        elif node_name == "budget":
                            collected["budget_md"] = update.get("budget_md", "")
                            st.markdown(collected["budget_md"] or "_Budget could not be calculated._")
                        elif node_name == "planner":
                            collected["itinerary"] = update.get("itinerary", "")
                            st.markdown(collected["itinerary"] or "_No itinerary generated._")
                        elif node_name == "memory":
                            st.markdown("🧠 **Preferences analyzed and updated in long-term memory!** ✨")

        # Save result in session state so subsequent interactions won't wipe the screen
        st.session_state.last_result = collected

    except Exception as e:
        st.error(f"Oops! Something went wrong during planning: {e}")
        st.stop()


# ── Render Saved Plan & Download (Persisted Across Reruns) ─────────────────────
if st.session_state.last_result:
    res = st.session_state.last_result

    # Display warnings / errors if any occurred
    if res.get("errors"):
        for err in res["errors"]:
            st.warning(f"⚠️ {err}")

    # Metrics
    status_icon = "⚠️" if res.get("errors") else "✅"
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box"><div class="metric-val">{res.get('agents_run', 0)}</div><div class="metric-lbl">Agents Run</div></div>
        <div class="metric-box"><div class="metric-val">{res.get('llm_calls', 0)}</div><div class="metric-lbl">LLM Calls</div></div>
        <div class="metric-box"><div class="metric-val">{status_icon}</div><div class="metric-lbl">Status</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Final Plan Display
    if res.get("itinerary"):
        st.markdown("<div class='sec-head'><span>📋 Final Travel Plan</span></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(res["itinerary"])

    # Prepare Markdown File Content
    now = datetime.now()
    filename = f"travel_plan_{now.strftime('%Y%m%d_%H%M%S')}.md"
    
    file_content = f"""# Travel Plan
**Query:** {res.get('query', '')}
**Generated:** {now.strftime("%Y-%m-%d %H:%M:%S")}
**User ID:** {res.get('user_id', user_id)}

---

## ✈️ Flight Research
{res.get('flight_results') or 'N/A'}

---

## 🏨 Hotel Research
{res.get('hotel_results') or 'N/A'}

---

## 🗓️ Itinerary & Budget
{res.get('itinerary') or 'N/A'}

---
*Generated by Multi-Agent Travel Planner • LLM Calls: {res.get('llm_calls', 0)}*
"""

    save_dir = BASE_DIR / "travel_plans"
    try:
        save_dir.mkdir(exist_ok=True)
        (save_dir / filename).write_text(file_content, encoding="utf-8")
        saved_locally = True
    except Exception:
        saved_locally = False

    # Responsive Download & Save info
    dl_col, info_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "⬇️ Download Plan (.md)",
            data=file_content,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
            key="download_plan_btn",
        )
    with info_col:
        if saved_locally:
            st.markdown(
                f"<div class='save-bar'>📁 Auto-saved → <code>travel_plans/{filename}</code></div>",
                unsafe_allow_html=True,
            )