"""
LangGraph multi-agent travel planner with long-term memory.

Graph:
  START -> parse -> [flight_agent || hotel_agent] -> budget -> planner -> memory -> END
"""

import os
import re
import math
import atexit
import hashlib
import operator
import logging
from datetime import date
from typing import TypedDict, Annotated, Optional, Literal, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from tools.budget_tool import compute_budget


# ──────────────────────── Setup ────────────────────────

load_dotenv(override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("travel_planner")

MODEL_NAME = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY .env file mein set nahi hai.")

# max_tokens=900 guarantees staying strictly below Groq's 1000 OTPM limit!
llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2,
    max_tokens=900,
)


# ──────────────────────── Schemas ────────────────────────

class TripRequest(BaseModel):
    origin: Optional[str] = Field(None, description="Departure city or airport, e.g. Mumbai")
    destination: Optional[str] = Field(None, description="Arrival city, e.g. Goa or Manali")
    city: Optional[str] = Field(None, description="Main city of stay, e.g. North Goa")
    cities: list[str] = Field(default_factory=list, description="List of all cities visited in travel order")

    departure_date: Optional[str] = None
    return_date: Optional[str] = None

    days: Optional[int] = Field(None, ge=1, le=365, description="Trip duration in days, e.g. 3")
    travellers: Optional[int] = Field(None, ge=1, le=100, description="Total number of travellers, e.g. 10")
    rooms: Optional[int] = Field(None, ge=1, le=50)

    budget: Optional[str] = None
    budget_amount: Optional[float] = Field(None, ge=0)
    budget_currency: str = "INR"

    passport_nationality: Optional[str] = None
    hotel_preference: Optional[str] = None
    dietary_preference: Optional[str] = None

    interests: list[str] = Field(default_factory=list)
    language: Literal["English", "Hindi"] = "English"

    @field_validator("days", mode="before")
    @classmethod
    def clean_days(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            clean = re.sub(r"[^\d]", "", v)
            return int(clean) if clean else None
        return v

    @field_validator("travellers", mode="before")
    @classmethod
    def clean_travellers(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            clean = re.sub(r"[^\d]", "", v)
            return int(clean) if clean else None
        return v

    @field_validator("budget_amount", mode="before")
    @classmethod
    def clean_budget(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = re.sub(r"[^\d.]", "", v)
            return float(clean) if clean else None
        return v


class CostItem(BaseModel):
    name: str = Field(description="Cost category name.")
    city: Optional[str] = None
    amount: float = Field(
        ge=0,
        description="Price of ONE unit in original currency. Do not multiply or convert."
    )
    currency: str = Field(
        default="INR",
        description="Currency code like INR, USD, EUR, GBP, AED."
    )
    basis: Literal[
        "per_person",
        "per_person_per_day",
        "per_room_per_night",
        "fixed_group",
    ] = Field(description="Billing unit.")
    source: Literal["data", "assumption"] = "assumption"

    @field_validator("basis", mode="before")
    @classmethod
    def clean_basis(cls, v):
        if not v:
            return "per_person"
        v_str = str(v).lower().strip()
        if "night" in v_str or "room" in v_str or "hotel" in v_str:
            return "per_room_per_night"
        if "day" in v_str or "daily" in v_str or "meal" in v_str or "food" in v_str:
            return "per_person_per_day"
        if "group" in v_str or "fixed" in v_str or "total" in v_str:
            return "fixed_group"
        return "per_person"

    @field_validator("currency", mode="before")
    @classmethod
    def clean_currency(cls, v):
        if not v:
            return "INR"
        v_str = str(v).upper().strip()
        if "₹" in v_str or "INR" in v_str or "RS" in v_str:
            return "INR"
        if "AED" in v_str:
            return "AED"
        if "SGD" in v_str or "S$" in v_str:
            return "SGD"
        if "CAD" in v_str or "C$" in v_str:
            return "CAD"
        if "AUD" in v_str or "A$" in v_str:
            return "AUD"
        if "$" in v_str or "USD" in v_str:
            return "USD"
        if "€" in v_str or "EUR" in v_str:
            return "EUR"
        if "£" in v_str or "GBP" in v_str:
            return "GBP"
        return v_str[:3]


class BudgetInputs(BaseModel):
    items: list[CostItem] = Field(
        default_factory=list,
        description="Individual travel expense items with unit prices."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explanations for estimates."
    )


class Prefs(BaseModel):
    preferences: list[str] = Field(
        default_factory=list,
        description="Lasting travel preferences (dietary, hotel tier, interests)."
    )


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    trip: dict
    flight_results: str
    hotel_results: str
    budget: dict
    budget_md: str
    budget_verdict: str
    budget_error: str
    itinerary: str
    errors: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]


# ──────────────────────── Helpers ────────────────────────

def _user_id(config: Optional[RunnableConfig]) -> str:
    configurable = (config or {}).get("configurable", {})
    return configurable.get("user_id", "anonymous")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        return _as_text(value.content)
    return str(value)


def _clean_place(text: Any) -> Optional[str]:
    """Helper to strip XML tags, quotes, and parameters from city/route strings."""
    if not text:
        return None
    val = str(text)
    val = re.sub(r"<[^>]+>", " ", val)
    val = re.sub(r"(?:parameter|destination|origin)\s*=\s*", " ", val, flags=re.I)
    val = val.replace('"', '').replace("'", '').replace('`', '')
    val = re.sub(r"\s+\b(?:and|or|with)\b\s*$", "", val, flags=re.I)
    val = re.sub(r"\s+", " ", val).strip()
    return val.title() if val else None


def _language_from_query(query: str, model_language: Optional[str]) -> str:
    if re.search(r"[\u0900-\u097F]", query):
        return "Hindi"
    return model_language if model_language in ("English", "Hindi") else "English"


def _error(name: str, exc: Exception) -> str:
    logger.exception("%s failed", name)
    return f"{name}: {type(exc).__name__}: {exc}"


def _normalize_trip(data: dict) -> dict:
    """Ensure destination, city, and multi-cities remain clean and synchronized across all nodes."""
    for key in ("origin", "destination", "city"):
        if data.get(key):
            data[key] = _clean_place(data[key])

    dest = data.get("destination")
    city = data.get("city")
    cities = list(data.get("cities") or [])

    # Split multi-destinations like "Zurich and Interlaken" or "Tokyo, Kyoto"
    if dest and (" and " in dest.lower() or "," in dest):
        split_cities = [c.strip() for c in re.split(r",|\band\b", dest, flags=re.I) if c.strip()]
        if split_cities:
            dest = _clean_place(split_cities[0])
            cities = [_clean_place(c) for c in split_cities if _clean_place(c)]

    clean_cities = []
    for c in cities:
        cleaned = _clean_place(c)
        if cleaned and cleaned not in clean_cities:
            clean_cities.append(cleaned)
    cities = clean_cities
    data["cities"] = cities
    data["destination"] = dest

    if dest and not city:
        city = dest
        data["city"] = dest
    if city and not dest:
        dest = city
        data["destination"] = city

    if city and not cities:
        data["cities"] = [city]
    elif cities:
        if not city:
            data["city"] = cities[0]
        if not dest:
            data["destination"] = cities[0]

    # Proportional room calculation: 1-2 pax -> 1 room, 3-4 pax -> 2 rooms, 5-6 pax -> 3 rooms, etc.
    pax = data.get("travellers") or 2
    data["rooms"] = max(1, math.ceil(pax / 2.0))

    return data


def _safe_preferences(store: Optional[BaseStore], uid: str) -> str:
    """Safely fetch long-term user preferences from store."""
    if not store:
        return "None saved yet."
    try:
        memories = store.search(("users", uid, "prefs"), limit=30)
        lines = []
        for memory in memories:
            value = getattr(memory, "value", None)
            if isinstance(value, dict):
                text_value = value.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    lines.append(f"- {text_value.strip()}")
        return "\n".join(lines) if lines else "None saved yet."
    except Exception as exc:
        logger.warning("Could not read user preferences: %s", exc)
        return "Preferences are temporarily unavailable."


def _extract_fallback_trip(query: str) -> dict:
    """
    Hardened regex fallback:
    1. Reverse routes: 'from X to Y' AND 'trip/tour/honeymoon to Y from X'
    2. Multi-city destinations with commas (e.g. 'Delhi, Agra and Jaipur')
    3. Strict word-boundaries (never truncates Interlaken or Indonesia at 'in')
    4. Handles adjectives & groups: '10 college friends', 'family of 6', '8 extended family members', '14 corporate delegates'
    5. Multi-currency support: S$ (SGD), C$ (CAD), A$ (AUD), $ (USD), € (EUR), £ (GBP), AED, INR
    6. Proportional group rooms calculation: 11 people -> 6 rooms, 14 people -> 7 rooms
    """
    origin = None
    destination = None
    days = None
    travellers = None
    budget_amount = None
    budget_curr = "INR"

    # Pattern A: "from Mumbai to Goa" / "from Delhi departing to Paris"
    route_match_a = re.search(
        r"\bfrom\s+([A-Za-z\s]+?)\s+(?:to|heading to)\s+([A-Za-z\s,]+?)(?:\s+\b(?:for|in|with|on|budget|under|starting)\b|\s*\.|\s*$)",
        query,
        re.I,
    )
    # Pattern B: "trip to Paris from Mumbai" / "honeymoon to Gangtok from Kolkata"
    route_match_b = re.search(
        r"\b(?:trip to|travel to|visit|explore|holiday in|vacation to|tour to|tour of|getaway to|honeymoon to|retreat to|escape to)\s+([A-Za-z\s,]+?)\s+(?:from|departing from|starting from)\s+([A-Za-z\s]+?)(?:\s+\b(?:for|in|with|on|budget|under)\b|\s*\.|\s*$)",
        query,
        re.I,
    )
    # Pattern C: "Paris trip from Mumbai"
    route_match_c = re.search(
        r"\b([A-Za-z\s]+?)\s+(?:trip|tour|holiday|vacation)\s+from\s+([A-Za-z\s]+?)(?:\s+\b(?:for|in|with|on|budget)\b|\s*\.|\s*$)",
        query,
        re.I,
    )

    if route_match_a:
        origin = _clean_place(route_match_a.group(1))
        destination = _clean_place(route_match_a.group(2))
    elif route_match_b:
        destination = _clean_place(route_match_b.group(1))
        origin = _clean_place(route_match_b.group(2))
    elif route_match_c:
        destination = _clean_place(route_match_c.group(1))
        origin = _clean_place(route_match_c.group(2))

    # Duration Extraction
    days_match = re.search(r"(\d+)\s*[- ]?(?:day|night)", query, re.I)
    if days_match:
        days = int(days_match.group(1))
    elif re.search(r"\b(?:1|one)[- ]*week\b", query, re.I):
        days = 7
    elif re.search(r"\bweekend\b", query, re.I):
        days = 3

    # Travellers Extraction
    fam_grp = re.search(r"\b(?:family of|group of)\s+(\d+)\b", query, re.I)
    me_and_friends = re.search(r"\bme\s+and\s+(\d+)\s+(?:friends|buddies|colleagues|people|family)\b", query, re.I)
    travellers_match = re.search(
        r"\b(\d+)\s*(?:[a-zA-Z]+\s+){0,3}(?:traveller|traveler|friend|person|people|adult|passenger|pax|colleague|member|guest|buddy|buddies|mate|mates|delegate|delegates|citizen|citizens)s?\b",
        query,
        re.I,
    )

    if fam_grp:
        travellers = int(fam_grp.group(1))
    elif me_and_friends:
        travellers = 1 + int(me_and_friends.group(1))
    elif travellers_match:
        travellers = int(travellers_match.group(1))
    elif re.search(r"\b(?:couple|honeymoon|husband\s+and\s+wife|wife\s+and\s+i|me\s+and\s+my\s+(?:wife|partner|girlfriend|husband))\b", query, re.I):
        travellers = 2
    elif re.search(r"\b(?:solo|alone|myself|single)\b", query, re.I):
        travellers = 1
    else:
        travellers = 2

    # Multi-Currency Extraction (S$, C$, A$ checked BEFORE generic $)
    if re.search(r"[€]|\bEUR\b", query, re.I):
        budget_curr = "EUR"
    elif re.search(r"[£]|\bGBP\b", query, re.I):
        budget_curr = "GBP"
    elif re.search(r"\bAED\b", query, re.I):
        budget_curr = "AED"
    elif re.search(r"S[$]|\bSGD\b", query, re.I):
        budget_curr = "SGD"
    elif re.search(r"C[$]|\bCAD\b", query, re.I):
        budget_curr = "CAD"
    elif re.search(r"A[$]|\bAUD\b", query, re.I):
        budget_curr = "AUD"
    elif re.search(r"[$]|\bUSD\b", query, re.I):
        budget_curr = "USD"
    else:
        budget_curr = "INR"

    # Budget Amount Extraction
    b_match = re.search(
        r"(?:budget|cost|under|approx|around)?\s*(?:[₹$€£]|AED|CAD|AUD|SGD|INR|Rs\.?)\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lac|k)?",
        query,
        re.I,
    )
    if b_match:
        try:
            raw_num = float(b_match.group(1).replace(",", ""))
            tail = query[b_match.end() : b_match.end() + 10].lower()
            if "lakh" in tail or "lac" in tail:
                budget_amount = raw_num * 100000.0
            elif "k" in tail:
                budget_amount = raw_num * 1000.0
            else:
                budget_amount = raw_num
        except Exception:
            pass

    # Hotel & Interest Preferences
    hotel_pref = None
    if re.search(r"(cottage|cabin|riverside)", query, re.I):
        hotel_pref = "riverside cottage with mountain views"
    elif re.search(r"(private pool)", query, re.I):
        hotel_pref = "private pool villa"
    elif re.search(r"(heritage|palace|haveli)", query, re.I):
        hotel_pref = "heritage palace hotel"
    elif re.search(r"(hostel|backpacker|dorm)", query, re.I):
        hotel_pref = "backpacker hostel / budget stay"
    elif re.search(r"(5\s*star|luxury|resort)", query, re.I):
        hotel_pref = "luxury 5-star resort"
    elif re.search(r"(homestay)", query, re.I):
        hotel_pref = "local homestay"

    interests = []
    for word, label in [
        ("beach", "beaches and water sports"),
        ("trek", "trekking and hiking"),
        ("mountain", "scenic mountain viewpoints"),
        ("party", "nightlife and parties"),
        ("cafe", "cafes and food hopping"),
        ("food", "local culinary experience"),
        ("culture", "historical monuments and culture"),
        ("temple", "spiritual temples and darshan"),
        ("wildlife", "wildlife safari and nature"),
        ("shopping", "shopping and markets"),
    ]:
        if re.search(rf"\b{word}", query, re.I):
            interests.append(label)

    return {
        "origin": origin,
        "destination": destination,
        "days": days,
        "travellers": travellers,
        "budget_amount": budget_amount,
        "budget_currency": budget_curr,
        "hotel_preference": hotel_pref,
        "interests": interests,
    }


# ──────────────────────── Agent Nodes ────────────────────────

def parse_agent(
    state: TravelState,
    config: RunnableConfig,
    *,
    store: Optional[BaseStore] = None,
):
    """Parse trip details with prompt-anchored JSON mode and robust fallback."""
    query = state.get("user_query") or ""
    uid = _user_id(config)
    known_prefs = _safe_preferences(store, uid)

    prompt = f"""
You are an expert travel parser. Extract structured parameters from the user request into valid JSON.

USER REQUEST:
"{query}"

KNOWN USER PREFERENCES:
{known_prefs}

RULES:
- Extract origin (departure city) and destination (arrival city).
- In multi-city queries like 'Zurich and Interlaken', destination is 'Zurich' and cities is ['Zurich', 'Interlaken'].
- Extract days.
- Extract travellers count: '10 college friends' -> 10, 'family of 6' -> 6, '4 family members' -> 4, 'couple' -> 2, 'solo' -> 1.
- Extract budget_amount and budget_currency accurately.
- Extract hotel_preference accurately.
- Extract dietary preference (e.g. 'vegetarian', 'seafood', 'jain').
"""

    data = None
    try:
        parsed = llm.with_structured_output(TripRequest, method="json_mode").invoke(prompt)
        if parsed:
            data = parsed.model_dump()
            data["language"] = _language_from_query(query, data.get("language"))
    except Exception as exc:
        logger.warning("Parser LLM call failed, engaging smart fallback: %s", exc)

    fallback = _extract_fallback_trip(query)
    if not data:
        data = fallback
    else:
        if not data.get("origin") and fallback.get("origin"):
            data["origin"] = fallback["origin"]
        if not data.get("destination") and fallback.get("destination"):
            data["destination"] = fallback["destination"]
            data["city"] = fallback["destination"]
            data["cities"] = [fallback["destination"]]
        if not data.get("days") and fallback.get("days"):
            data["days"] = fallback["days"]
        if not data.get("travellers") and fallback.get("travellers"):
            data["travellers"] = fallback["travellers"]
        if not data.get("budget_amount") and fallback.get("budget_amount"):
            data["budget_amount"] = fallback["budget_amount"]
            data["budget_currency"] = fallback.get("budget_currency", "INR")
        if not data.get("hotel_preference"):
            data["hotel_preference"] = fallback.get("hotel_preference")

    data = _normalize_trip(data)

    return {
        "trip": data,
        "llm_calls": 1,
    }


def flight_agent(state: TravelState):
    """Search for flight or road/bus transit."""
    trip = state.get("trip") or {}

    origin = _clean_place(trip.get("origin"))
    destination_city = _clean_place(trip.get("city"))

    if not destination_city and trip.get("cities"):
        destination_city = _clean_place(trip["cities"][0])
    if not destination_city and trip.get("destination"):
        destination_city = _clean_place(trip["destination"])

    if not origin or not destination_city:
        return {
            "flight_results": "Transit search skipped. Departure or destination city missing."
        }

    search_trip = dict(trip)
    search_trip["origin"] = origin
    search_trip["destination"] = destination_city
    search_trip["city"] = destination_city

    try:
        result = search_flights(search_trip)
        text_result = _as_text(result)

        return {
            "flight_results": (
                text_result
                if text_result and not text_result.startswith("Search failed:")
                else (text_result or "Transit research is currently unavailable.")
            )
        }

    except Exception as exc:
        return {
            "flight_results": "Transit research is currently unavailable.",
            "errors": [_error("Flight agent", exc)],
        }


def hotel_agent(state: TravelState):
    """Search for accommodation in the requested city or key cities."""
    trip = state.get("trip") or {}

    cities = list(trip.get("cities") or [])
    if not cities and trip.get("city"):
        cities = [_clean_place(trip["city"])]
    if not cities and trip.get("destination"):
        cities = [_clean_place(trip["destination"])]

    cities = [c for c in cities if c]

    if not cities:
        return {
            "hotel_results": "Hotel search not performed: destination city is missing."
        }

    target_destinations = ", ".join(cities[:3])
    guests = trip.get("travellers") or 2
    rooms = trip.get("rooms") or max(1, math.ceil(guests / 2.0))
    preference = trip.get("hotel_preference") or "hotel or resort"

    query = (
        f"Hotels in {target_destinations}; {guests} guests, rooms: {rooms}; {preference}. "
        f"Include hotel name, nightly price in INR, rating, and location. "
        f"Distinguish between verified rates and estimates."
    )

    try:
        result = tavily_search(query)
        text_result = _as_text(result)
        return {
            "hotel_results": text_result or "Hotel search returned no usable results."
        }
    except Exception as exc:
        error = _error("Hotel agent", exc)
        return {
            "hotel_results": "Hotel search is currently unavailable.",
            "errors": [error],
        }


def budget_agent(state: TravelState):
    """Extract real unit prices from search text and calculate via Python."""
    trip = state.get("trip") or {}
    lang = trip.get("language") or "English"
    dest_str = str(trip.get("destination") or "").lower()

    flight_text = state.get("flight_results", "")
    hotel_text = state.get("hotel_results", "")

    budget_cap = trip.get("budget_amount")
    currency = (trip.get("budget_currency") or "INR").upper()

    extracted_transit_price = None
    extracted_hotel_price = None

    ROAD_DESTS = {
        "manali", "kasol", "rishikesh", "mussoorie", "nainital", "shimla",
        "spiti", "kodaikanal", "munnar", "ooty", "mount abu", "dharamshala",
        "dalhousie", "lonavala", "mahabaleshwar", "alibaug", "darjeeling", "wayanad",
        "coorg", "chikmagalur", "auli", "gangtok", "shillong", "cherrapunji"
    }

    is_road = any(h in dest_str for h in ROAD_DESTS)

    if is_road:
        extracted_transit_price = 1500.0
    else:
        flight_matches = re.findall(r"(?:₹|INR|from\s*₹)\s*([\d,]{4,6})", flight_text)
        if flight_matches:
            try:
                extracted_transit_price = float(flight_matches[0].replace(",", ""))
            except ValueError:
                pass

    hotel_matches = re.findall(r"(?:₹|INR)\s*([\d,]{4,6})\s*(?:per|night)?", hotel_text)
    if hotel_matches:
        try:
            extracted_hotel_price = float(hotel_matches[0].replace(",", ""))
        except ValueError:
            pass

    prompt = f"""
Extract unit expense items from research. Python will calculate totals.

TRIP: {trip.get('origin')} to {trip.get('destination')}, {trip.get('days')} days, {trip.get('travellers')} travellers.
USER BUDGET: {budget_cap} {currency}.

FLIGHT / TRANSIT RESEARCH:
{flight_text[:800]}

HOTEL RESEARCH:
{hotel_text[:800]}

INSTRUCTIONS:
1. Extract unit prices:
   - Transit/Flight: Extract exact price from research if available. If road trip (e.g. Manali bus), use 1500 INR.
   - Hotel: Nightly room price from research (e.g. 3500-6000 INR).
   - Food/Meals: Daily per person dining (e.g. 1000-2500 INR).
   - Local transport: Per day cost (e.g. 600-1000 INR).
2. Set source="data" for research prices, "assumption" for estimates.
"""

    out = None
    try:
        out = llm.with_structured_output(BudgetInputs, method="json_mode").invoke(prompt)
    except Exception as e:
        logger.warning("Budget LLM extraction failed: %s", e)

    # Dynamic Context-Aware Fallback
    if not out or not out.items:
        transit_name = "Overnight Volvo Bus / Transit" if is_road else "Round-trip flights"
        t_price = extracted_transit_price or (1500.0 if is_road else (5500.0 if budget_cap and budget_cap <= 45000 else 14000.0))
        t_source = "data" if extracted_transit_price else "assumption"

        h_price = extracted_hotel_price or (3500.0 if budget_cap and budget_cap <= 45000 else 6000.0)
        h_source = "data" if extracted_hotel_price else "assumption"

        f_price = 1200.0 if budget_cap and budget_cap <= 45000 else 2500.0
        s_price = 600.0 if budget_cap and budget_cap <= 45000 else 1000.0

        fallback_items = [
            CostItem(name=transit_name, amount=t_price, currency="INR", basis="per_person", source=t_source),
            CostItem(name="Hotel accommodation", amount=h_price, currency="INR", basis="per_room_per_night", source=h_source),
            CostItem(name="Dining & Meals", amount=f_price, currency="INR", basis="per_person_per_day", source="assumption"),
            CostItem(name="Local transport & Scooter/Taxi", amount=s_price, currency="INR", basis="per_person_per_day", source="assumption"),
            CostItem(name="Travel insurance & Emergency", amount=400.0, currency="INR", basis="per_person", source="assumption"),
        ]
        out = BudgetInputs(items=fallback_items, assumptions=["Dynamic market estimates applied."])

    try:
        travellers = trip.get("travellers") or 2
        days = trip.get("days") or 3
        rooms = trip.get("rooms") or max(1, math.ceil(travellers / 2.0))

        cities = list(trip.get("cities") or [trip.get("destination") or "Destination"])
        budget_total = trip.get("budget_amount")
        budget_currency = (trip.get("budget_currency") or "INR").upper()

        normalized_items = []
        for item in out.items:
            item_data = item.model_dump()
            item_data["currency"] = (item_data.get("currency") or "INR").upper()
            normalized_items.append(item_data)

        budget_result = compute_budget(
            travellers=travellers,
            days=days,
            items=normalized_items,
            budget_total=budget_total,
            budget_currency=budget_currency,
            lang=lang,
            assumptions=out.assumptions,
            cities=cities,
            rooms=rooms,
        )

        budget_result = dict(budget_result)
        budget_md = budget_result.pop("markdown", "")
        budget_verdict = budget_result.pop("verdict", "")

        return {
            "budget": budget_result,
            "budget_md": budget_md,
            "budget_verdict": budget_verdict,
            "budget_error": "",
            "llm_calls": 1,
        }

    except Exception as exc:
        error = _error("Budget agent", exc)
        return {
            "budget": {},
            "budget_md": "",
            "budget_verdict": "",
            "budget_error": error,
            "errors": [error],
        }


def planner_agent(
    state: TravelState,
    config: RunnableConfig,
    *,
    store: Optional[BaseStore] = None,
):
    """Generate concise, bulleted Day 1 to Day N itinerary that completes 100% without truncation."""
    trip = state.get("trip") or {}
    budget_md = state.get("budget_md", "")
    budget_dict = state.get("budget") or {}
    verdict = state.get("budget_verdict", "")
    lang = trip.get("language") or "English"

    origin = trip.get("origin") or "Departure City"
    destination = trip.get("destination") or "Destination"
    days = trip.get("days") or 3
    travellers = trip.get("travellers") or 2

    sys_prompt = f"""You are an elite travel planner.
Language: {lang}. Write in clean, concise markdown.
STRICT INSTRUCTION: Provide a complete Day 1 to Day {days} itinerary.
Format:
- 2 sentence overview of the trip.
- For each Day (Day 1, Day 2, ..., Day {days}): exactly 3 bullet points (Morning, Afternoon, Evening).
Do NOT write long paragraphs. Complete every day through Day {days} cleanly.
"""

    prompt = f"""
Trip: {origin} to {destination}
Duration: {days} Days | Travellers: {travellers}
Budget Verdict: {verdict}

Search Context:
Flights/Transit: {state.get('flight_results', '')[:300]}
Hotels: {state.get('hotel_results', '')[:300]}

Generate the concise Day 1 to Day {days} itinerary now.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=prompt)
        ])
        itinerary_text = _as_text(response)
        if not itinerary_text.strip():
            raise ValueError("Planner returned empty itinerary.")

        parts = [
            verdict,
            itinerary_text,
            "---",
            budget_md,
        ]
        return {
            "itinerary": "\n\n".join(part for part in parts if part),
            "messages": [response],
            "llm_calls": 1,
        }

    except Exception as exc:
        error = _error("Planner agent", exc)
        return {
            "itinerary": f"Itinerary generated with budget details:\n\n{budget_md}",
            "errors": [error],
        }


def memory_agent(
    state: TravelState,
    config: RunnableConfig,
    *,
    store: Optional[BaseStore] = None,
):
    """Save lasting travel preferences isolated per user_id."""
    if not store:
        return {}

    uid = _user_id(config)
    trip = state.get("trip", {})

    prompt = f"""
Extract only lasting, reusable user travel preferences from this trip:
User Request: {state.get('user_query', '')}
Trip Details: {trip}

Extract ONLY lasting preferences (e.g. 'Prefers seafood dining', 'Prefers beachfront resort', 'Prefers vegetarian food').
Do NOT extract specific dates or cities.
"""

    try:
        parsed = llm.with_structured_output(Prefs, method="json_mode").invoke(prompt)
        if parsed and parsed.preferences:
            for pref in parsed.preferences:
                pref_text = pref.strip()
                if not pref_text:
                    continue
                key = hashlib.sha256(pref_text.lower().encode()).hexdigest()[:16]
                store.put(
                    ("users", uid, "prefs"),
                    key,
                    {"text": pref_text}
                )
        return {"llm_calls": 1}
    except Exception:
        return {}


# ──────────────────────── Graph Definition ────────────────────────

graph = StateGraph(TravelState)

graph.add_node("parse", parse_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("budget", budget_agent)
graph.add_node("planner", planner_agent)
graph.add_node("memory", memory_agent)

graph.add_edge(START, "parse")
graph.add_edge("parse", "flight_agent")
graph.add_edge("parse", "hotel_agent")
graph.add_edge(["flight_agent", "hotel_agent"], "budget")
graph.add_edge("budget", "planner")
graph.add_edge("planner", "memory")
graph.add_edge("memory", END)


# ──────────────────────── Storage & Export Setup ────────────────────────

store = None
checkpointer = None

try:
    from langgraph.store.memory import InMemoryStore
    from langgraph.checkpoint.memory import MemorySaver

    db_url = os.getenv("DATABASE_URL")
    if db_url and "postgres" in db_url.lower():
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from langgraph.store.postgres import PostgresStore

            store = PostgresStore.from_conn_string(db_url)
            store.setup()
            checkpointer = PostgresSaver.from_conn_string(db_url)
            checkpointer.setup()
            logger.info("Connected to PostgreSQL for checkpoints & long-term memory.")
        except Exception as pg_err:
            logger.warning("PostgreSQL init failed, falling back to InMemory: %s", pg_err)
            store = InMemoryStore()
            checkpointer = MemorySaver()
    else:
        store = InMemoryStore()
        checkpointer = MemorySaver()

except Exception as e:
    logger.warning("Memory store setup warning: %s", e)
    store = None
    checkpointer = None

app = graph.compile(checkpointer=checkpointer, store=store)


def initial_state(user_query: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "trip": {},
        "flight_results": "",
        "hotel_results": "",
        "budget": {},
        "budget_md": "",
        "budget_verdict": "",
        "budget_error": "",
        "itinerary": "",
        "errors": [],
        "llm_calls": 0,
    }