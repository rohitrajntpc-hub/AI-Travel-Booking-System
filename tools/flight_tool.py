import os
import re
import logging
from typing import Optional
import requests
from dotenv import load_dotenv
from tools.tavily_tool import tavily_search

load_dotenv()
logger = logging.getLogger(__name__)

AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_API_KEY")

# Complete list of 42+ hill stations & road destinations without direct passenger airports
ROAD_OR_TRAIN_DESTINATIONS = {
    "manali", "kasol", "rishikesh", "mussoorie", "nainital", "shimla",
    "spiti", "kodaikanal", "munnar", "ooty", "mount abu", "dharamshala",
    "dalhousie", "lonavala", "mahabaleshwar", "alibaug", "darjeeling", "wayanad",
    "coorg", "chikmagalur", "auli", "gangtok", "shillong", "cherrapunji",
    "tawang", "pachmarhi", "jim corbett", "jibhi", "kufri", "valparai",
    "yercaud", "panchgani", "dhanaulti", "kalimpong", "kudremukh",
    "lansdowne", "kinnaur", "kasauli", "sakleshpur", "vagamon", "bhimtal", "shivpuri",
    "haridwar", "hampi"
}


def _get_aviationstack_routes(origin: str, destination: str) -> Optional[str]:
    """
    Query AviationStack for airline route schedules, strictly keeping only
    flights that actually match the origin or destination.
    """
    if not AVIATIONSTACK_KEY:
        return None

    try:
        url = "http://api.aviationstack.com/v1/flights"
        params = {
            "access_key": AVIATIONSTACK_KEY,
            "limit": 10,
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            flights = data.get("data", [])
            if flights and isinstance(flights, list):
                schedules = []
                orig_lower = origin.lower()
                dest_lower = destination.lower()

                for f in flights:
                    airline = (f.get("airline") or {}).get("name", "Commercial Airline")
                    flight_num = (f.get("flight") or {}).get("iata", "N/A")
                    dep_airport = (f.get("departure") or {}).get("airport", "")
                    arr_airport = (f.get("arrival") or {}).get("airport", "")

                    # Strictly filter: flight must relate to origin or destination
                    match_origin = orig_lower in dep_airport.lower()
                    match_dest = dest_lower in arr_airport.lower()

                    if match_origin or match_dest:
                        schedules.append(f"- **{airline}** ({flight_num}): {dep_airport} → {arr_airport}")
                        if len(schedules) >= 3:
                            break

                if schedules:
                    return "### 📡 Live Airline Route Schedule (AviationStack)\n" + "\n".join(schedules)
    except Exception as exc:
        logger.debug("AviationStack lookup skipped or failed: %s", exc)

    return None


def search_flights(trip: dict) -> str:
    """
    Search transit/flight information combining AviationStack route data and Tavily web research.
    Intelligently routes hill stations (e.g. Manali, Ooty, Rishikesh, Munnar) to Volvo bus / cab.
    """
    if not isinstance(trip, dict):
        return "Transit research failed: Invalid trip details."

    origin = str(trip.get("origin") or "").strip()
    destination = str(trip.get("destination") or "").strip()

    cities = trip.get("cities") or []
    if isinstance(cities, str):
        cities = [cities]

    entry_city = ""
    if cities:
        clean_cities = [str(c).strip() for c in cities if str(c).strip()]
        if clean_cities:
            entry_city = clean_cities[0]

    target_destination = destination or entry_city

    if not origin:
        return "Flight search skipped: Departure city is missing."

    if not target_destination:
        return "Flight search skipped: Destination city is missing."

    if origin.casefold() == target_destination.casefold():
        return f"Flight search skipped: Origin and destination are the same ({origin})."

    departure_date = str(trip.get("departure_date") or "").strip()
    return_date = str(trip.get("return_date") or "").strip()

    has_dep_date = bool(departure_date and departure_date.lower() != "not specified")
    has_ret_date = bool(return_date and return_date.lower() != "not specified")

    days = trip.get("days")
    try:
        days_count = int(days) if days is not None else None
    except (TypeError, ValueError):
        days_count = None

    travellers = trip.get("travellers", 2)
    try:
        travellers = max(1, min(int(travellers), 100))
    except (TypeError, ValueError):
        travellers = 2

    is_round_trip = has_ret_date or (days_count is not None and days_count > 1)

    if has_dep_date and has_ret_date:
        journey_type = f"round trip, departure {departure_date}, return {return_date}"
    elif has_dep_date and is_round_trip:
        journey_type = f"round trip, departing {departure_date} (~{days_count or 4} days stay)"
    elif is_round_trip:
        journey_type = f"round trip (~{days_count or 4} days duration, dates flexible)"
    else:
        journey_type = "dates flexible / approximate"

    # Substring check for hill stations and road destinations
    dest_lower = target_destination.lower()
    is_road_trip = any(rd in dest_lower for rd in ROAD_OR_TRAIN_DESTINATIONS)

    if is_road_trip:
        query = (
            f"How to travel from {origin} to {target_destination} {journey_type} for {travellers} people. "
            f"Overnight Volvo bus fares, private cab prices, nearest railway station or airport route options."
        )
    else:
        query = (
            f"Flights {origin} to {target_destination} {journey_type} for {travellers} passengers. "
            f"Airlines, approximate economy roundtrip fares in INR, duration, baggage allowance. "
            f"Official airline booking links."
        )

    aviation_data = None if is_road_trip else _get_aviationstack_routes(origin, target_destination)

    try:
        result = tavily_search(query=query, max_results=5)

        if not result or not isinstance(result, str):
            return "Transit research failed: No usable search results."

        if result.startswith("Search failed:") or result.startswith("Search unavailable:"):
            return result

        header = "## 🚌 Road & Transit Research Snippets" if is_road_trip else "## ✈️ Flight Research Snippets"
        transit_note = ""
        if is_road_trip:
            transit_note = f"💡 **Route Note:** {target_destination} has no commercial passenger airport. Travel is typically via overnight Volvo bus, cab from nearest railway/airport junction, or personal vehicle."

        output_parts = [
            header,
            f"**Route:** {origin} → {target_destination}",
            f"**Passengers/Travellers:** {travellers}",
            f"**Journey:** {journey_type}",
            transit_note,
        ]

        if aviation_data:
            output_parts.append(aviation_data + "\n")

        output_parts.extend([
            "⚠️ **Important:** Web research snippets reflect estimates. Verify current fares directly with travel operators.\n",
            result,
        ])

        return "\n\n".join(part for part in output_parts if part)

    except Exception:
        logger.exception("Transit search failed")
        return "Transit research failed due to an unexpected error."