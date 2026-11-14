"""
Reliable travel budget calculator.

LLM supplies expense items and unit prices.
Python handles quantities, city-wise hotel nights, currency conversion,
contingency, totals and budget validation.
"""

import math
import time
import logging
import requests

logger = logging.getLogger("travel_planner.budget_tool")

FALLBACK_RATES = {
    "USD": 1.0,
    "INR": 88.0,
    "AED": 3.6725,
    "EUR": 0.92,
    "GBP": 0.78,
    "THB": 35.0,
    "JPY": 150.0,
    "SGD": 1.30,
    "IDR": 16000.0,
    "MYR": 4.5,
    "AUD": 1.55,
    "CAD": 1.38,
    "CHF": 0.88,
    "NZD": 1.70,
    "VND": 25400.0,
    "KRW": 1390.0,
    "CNY": 7.25,
    "TRY": 34.5,
    "SAR": 3.75,
    "QAR": 3.64,
    "NPR": 140.0,
    "LKR": 290.0,
    "MYR": 4.45,
}

CURRENCY_ALIASES = {
    "₹": "INR",
    "RS": "INR",
    "RS.": "INR",
    "RUPEE": "INR",
    "RUPEES": "INR",
    "$": "USD",
    "USD$": "USD",
    "DOLLAR": "USD",
    "DOLLARS": "USD",
    "€": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    "£": "GBP",
    "POUND": "GBP",
    "POUNDS": "GBP",
    "DIRHAM": "AED",
    "DIRHAMS": "AED",
    "DHS": "AED",
    "DH": "AED",
    "BAHT": "THB",
    "YEN": "JPY",
    "YEN": "JPY",
    "S$": "SGD",
    "SGD$": "SGD",
}

VALID_BASES = {
    "per_person",
    "per_person_per_day",
    "per_room_per_night",
    "per_group",
    "per_group_per_day",
}

CONTINGENCY_PCT = 5

_cache = {
    "t": 0.0,
    "rates": None,
    "live": False,
}

LABELS = {
    "English": {
        "title": "Budget breakdown (calculated by Python)",
        "item": "Item",
        "unit": "Unit price",
        "qty": "Quantity",
        "total": "Total",
        "source": "Source",
        "data": "search data",
        "assumption": "assumption",
        "travellers": "travellers",
        "traveller": "traveller",
        "days": "days",
        "nights": "nights",
        "rooms": "rooms",
        "group": "whole group",
        "subtotal": "Subtotal",
        "contingency": f"Contingency ({CONTINGENCY_PCT}%)",
        "grand": "Grand total",
        "rate": "Exchange rate",
        "live": "live rate",
        "fallback": "approximate fallback rate; verify before booking",
        "trip": "Trip",
        "route": "Route",
        "per_person": "Per person",
        "thin": (
            "Only {pct}% of your budget remains. "
            "Many prices are assumptions; verify before booking."
        ),
        "assump_share": "Share based on assumptions",
        "assumptions": "Assumptions",
        "skipped": "Skipped items",
        "within": (
            "✅ **Budget check:** estimated total {total} "
            "against budget {budget}. {left} left."
        ),
        "over": (
            "⚠️ **Budget check:** estimated total {total} "
            "is {gap} above your budget {budget}."
        ),
        "nobudget": "ℹ️ Estimated total: {total} (no budget provided).",
        "city_nights": "City-wise hotel nights",
    },
    "Hindi": {
        "title": "बजट का हिसाब (Python से calculate किया गया)",
        "item": "मद",
        "unit": "इकाई मूल्य",
        "qty": "मात्रा",
        "total": "कुल",
        "source": "स्रोत",
        "data": "सर्च डेटा",
        "assumption": "अनुमान",
        "travellers": "यात्री",
        "traveller": "यात्री",
        "days": "दिन",
        "nights": "रातें",
        "rooms": "कमरे",
        "group": "पूरा समूह",
        "subtotal": "उप-योग",
        "contingency": f"अतिरिक्त खर्च ({CONTINGENCY_PCT}%)",
        "grand": "कुल योग",
        "rate": "विनिमय दर",
        "live": "लाइव दर",
        "fallback": "अनुमानित दर; बुकिंग से पहले जाँचें",
        "trip": "यात्रा",
        "route": "मार्ग",
        "per_person": "प्रति व्यक्ति",
        "thin": (
            "आपके बजट का सिर्फ़ {pct}% बचा है। "
            "बुकिंग से पहले अनुमानित कीमतें जाँचें।"
        ),
        "assump_share": "अनुमान पर आधारित खर्च",
        "assumptions": "मान्यताएँ",
        "skipped": "छोड़े गए खर्च",
        "within": (
            "✅ **बजट जाँच:** अनुमानित कुल {total}, "
            "बजट {budget}। बचत {left}।"
        ),
        "over": (
            "⚠️ **बजट जाँच:** अनुमानित कुल {total}, "
            "आपके बजट {budget} से {gap} ज़्यादा।"
        ),
        "nobudget": "ℹ️ अनुमानित कुल {total} (बजट नहीं बताया गया)।",
        "city_nights": "शहर के हिसाब से होटल की रातें",
    },
}


def _clean_cur(cur):
    value = str(cur or "INR").strip().upper()
    return CURRENCY_ALIASES.get(value, value)


def _safe_number(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for {name}: {value}")

    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")

    return number


def get_rates():
    """
    Returns USD-base exchange rates:
    (rates_dict, is_live)
    """
    now = time.time()
    # Cache live rates for 1 hour
    if _cache["rates"] and (now - _cache["t"] < 3600) and _cache["live"]:
        return _cache["rates"], _cache["live"]

    try:
        response = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            timeout=6,
        )
        response.raise_for_status()

        data = response.json()
        rates = data.get("rates", {})

        if data.get("result") == "success":
            cleaned_rates = dict(FALLBACK_RATES)  # Keep fallbacks as baseline

            for currency, rate in rates.items():
                try:
                    r = float(rate)
                    if math.isfinite(r) and r > 0:
                        cleaned_rates[currency.upper()] = r
                except (TypeError, ValueError):
                    continue

            if cleaned_rates.get("USD") == 1.0 and cleaned_rates.get("INR", 0) > 0:
                _cache.update(t=now, rates=cleaned_rates, live=True)
                return cleaned_rates, True

    except Exception as exc:
        logger.warning("Exchange rate API unavailable (%s). Using fallback rates.", exc)

    # Use fallback rates without long caching so we can retry on next run
    fallback = dict(FALLBACK_RATES)
    _cache.update(t=now, rates=fallback, live=False)
    return fallback, False


def fmt_inr(value):
    """Indian number formatting: ₹2,13,290."""
    number = int(round(value))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))

    if len(digits) > 3:
        head = digits[:-3]
        tail = digits[-3:]
        parts = []

        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]

        if head:
            parts.insert(0, head)

        digits = ",".join(parts + [tail])

    return f"{sign}₹{digits}"


def _fmt_amt(value):
    value = float(value)
    if value.is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _reconcile_city_nights(city_nights, total_nights, cities=None):
    """
    Reconciles city-night mapping safely without raising fatal exceptions.
    Allocates missing nights or normalizes if user-provided nights do not match total_nights.
    """
    if total_nights <= 0:
        return {}, []

    notes = []
    result = {}

    if isinstance(city_nights, dict):
        for city, nights in city_nights.items():
            city_name = str(city).strip()
            if not city_name:
                continue
            try:
                nights_val = int(round(float(nights)))
                if nights_val >= 0:
                    result[city_name.casefold()] = nights_val
            except (ValueError, TypeError):
                continue

    allocated = sum(result.values())

    # If no city_nights specified, distribute across known cities if available
    if not result and cities:
        city_count = len(cities)
        base_nights = max(1, total_nights // city_count)
        for i, c in enumerate(cities):
            c_key = str(c).strip().casefold()
            # Assign remainder to the first city
            result[c_key] = base_nights + (total_nights - base_nights * city_count if i == 0 else 0)
        return result, notes

    # If partial allocation was provided
    if result and allocated != total_nights:
        diff = total_nights - allocated
        notes.append(
            f"Specified hotel nights ({allocated}) differed from total trip nights ({total_nights}). Adjusted automatically."
        )
        if diff > 0:
            # Distribute remaining nights to the last city
            last_key = list(result.keys())[-1]
            result[last_key] += diff
        elif diff < 0:
            # Scale down proportionally
            scale = total_nights / max(1, allocated)
            scaled = {k: max(1, int(round(v * scale))) for k, v in result.items()}
            # Adjust rounding difference
            diff_scaled = total_nights - sum(scaled.values())
            first_key = list(scaled.keys())[0]
            scaled[first_key] = max(1, scaled[first_key] + diff_scaled)
            result = scaled

    return result, notes


def compute_budget(
    *,
    travellers,
    days,
    items,
    budget_total=None,
    budget_currency="INR",
    lang="English",
    assumptions=None,
    cities=None,
    rooms=None,
    city_nights=None,
):
    """
    Calculate travel expenses.

    Supported item fields:
        name, amount, currency, basis, source, city (optional)
    """
    L = LABELS["Hindi" if lang == "Hindi" else "English"]
    assumptions_list = list(assumptions or [])

    travellers_value = _safe_number(travellers, "travellers")
    days_value = _safe_number(days, "days")

    travellers = max(1, min(int(round(travellers_value)), 100))
    days = max(1, min(int(round(days_value)), 365))
    nights = max(days - 1, 1 if days > 1 else 0)

    if rooms is None:
        rooms = max(1, math.ceil(travellers / 2))
    else:
        rooms_value = _safe_number(rooms, "rooms")
        rooms = max(1, min(int(round(rooms_value)), 50))

    rates, live = get_rates()
    inr_per_usd = rates.get("INR", FALLBACK_RATES["INR"])

    # Reconcile city nights safely without crashing
    city_nights_map, night_notes = _reconcile_city_nights(city_nights, nights, cities)
    assumptions_list.extend(night_notes)

    tw = L["traveller"] if travellers == 1 else L["travellers"]

    qty_map = {
        "per_person": (travellers, f"{travellers} {tw}"),
        "per_person_per_day": (travellers * days, f"{travellers} {tw} × {days} {L['days']}"),
        "per_room_per_night": (rooms * nights, f"{rooms} {L['rooms']} × {nights} {L['nights']}"),
        "per_group": (1, L["group"]),
        "per_group_per_day": (days, f"{days} {L['days']}"),
    }

    lines = []
    skipped = []

    for item in items or []:
        if not isinstance(item, dict):
            skipped.append("Invalid expense item")
            continue

        name = str(item.get("name", "Unnamed expense"))
        name = name.replace("|", "/").replace("\n", " ").strip()

        currency = _clean_cur(item.get("currency", "INR"))
        basis = str(item.get("basis", "")).strip()

        try:
            amount = _safe_number(item.get("amount", 0), name)
        except ValueError:
            skipped.append(name)
            continue

        if basis not in VALID_BASES or currency not in rates or amount < 0:
            skipped.append(name)
            continue

        quantity, quantity_text = qty_map[basis]
        city = str(item.get("city", "") or "").replace("|", "/").strip()

        # City-specific hotel calculation
        if basis == "per_room_per_night":
            if nights == 0:
                # 1-day trip without overnight stay
                quantity = rooms
                quantity_text = f"{rooms} {L['rooms']} (Day use)"
            elif city:
                city_key = city.casefold()
                city_stay_nights = city_nights_map.get(city_key)

                if city_stay_nights is None:
                    # Allocate fallback nights instead of skipping
                    city_stay_nights = max(1, nights // max(1, len(cities or [city])))

                quantity = rooms * city_stay_nights
                quantity_text = f"{rooms} {L['rooms']} × {city_stay_nights} {L['nights']} ({city})"

        unit_inr = amount * inr_per_usd / rates[currency]
        total_inr = int(round(unit_inr * quantity))
        source_is_data = str(item.get("source", "")).strip().lower() == "data"

        lines.append({
            "name": name,
            "unit": f"{_fmt_amt(amount)} {currency}",
            "qty_text": quantity_text,
            "total": total_inr,
            "source": L["data"] if source_is_data else L["assumption"],
            "is_data": source_is_data,
        })

    if not lines:
        raise ValueError("No valid budget items found. Check expense amount, currency and basis.")

    subtotal = sum(line["total"] for line in lines)
    data_total = sum(line["total"] for line in lines if line["is_data"])
    assump_share = round(100 * (subtotal - data_total) / subtotal) if subtotal > 0 else 0

    contingency = int(round(subtotal * CONTINGENCY_PCT / 100))
    total = subtotal + contingency

    # Convert user's budget into INR
    budget_inr = None
    if budget_total is not None:
        try:
            budget_amount = _safe_number(budget_total, "budget_total")
            budget_cur = _clean_cur(budget_currency)
            if budget_amount >= 0 and budget_cur in rates:
                budget_inr = int(round(budget_amount * inr_per_usd / rates[budget_cur]))
        except ValueError:
            budget_inr = None

    if budget_inr is None:
        verdict = L["nobudget"].format(total=fmt_inr(total))
        within = None
        diff = None
    elif total <= budget_inr:
        diff = budget_inr - total
        within = True
        verdict = L["within"].format(
            total=fmt_inr(total),
            budget=fmt_inr(budget_inr),
            left=fmt_inr(diff),
        )
    else:
        diff = total - budget_inr
        within = False
        verdict = L["over"].format(
            total=fmt_inr(total),
            budget=fmt_inr(budget_inr),
            gap=fmt_inr(diff),
        )

    # Thin margin check with zero-division guard
    if within and budget_inr and budget_inr > 0 and (budget_inr - total) / budget_inr < 0.15:
        remaining_pct = round(100 * (budget_inr - total) / budget_inr)
        verdict += " " + L["thin"].format(pct=remaining_pct)

    rate_note = f"1 USD = ₹{inr_per_usd:.2f} ({L['live'] if live else L['fallback']})"

    md = [
        f"### 💰 {L['title']}",
        f"{L['rate']}: {rate_note}",
        f"{L['trip']}: {travellers} {tw} · {days} {L['days']} · {nights} {L['nights']} · {rooms} {L['rooms']}",
    ]

    if cities:
        md.append(f"{L['route']}: {' → '.join(map(str, cities))}")

    if city_nights_map and len(city_nights_map) > 1:
        md.append("")
        md.append(f"**{L['city_nights']}:**")
        for city_k, count in city_nights_map.items():
            md.append(f"- {city_k.title()}: {count} {L['nights']}")

    md += [
        "",
        f"| {L['item']} | {L['unit']} | {L['qty']} | {L['total']} (₹) | {L['source']} |",
        "|---|---|---|---:|---|",
    ]

    for line in lines:
        md.append(
            f"| {line['name']} | {line['unit']} | {line['qty_text']} | {fmt_inr(line['total'])} | {line['source']} |"
        )

    md += [
        f"| **{L['subtotal']}** | | | **{fmt_inr(subtotal)}** | |",
        f"| {L['contingency']} | | | {fmt_inr(contingency)} | |",
        f"| **{L['grand']}** | | | **{fmt_inr(total)}** | |",
    ]

    if travellers > 1:
        md += ["", f"{L['per_person']}: **{fmt_inr(total / travellers)}**"]

    md += ["", f"{L['assump_share']}: **{assump_share}%**"]

    if assumptions_list:
        md += ["", f"**{L['assumptions']}:**"]
        md += [f"- {str(a)}" for a in assumptions_list]

    if skipped:
        md += ["", f"_{L['skipped']}: {', '.join(skipped)}_"]

    return {
        "travellers": travellers,
        "days": days,
        "nights": nights,
        "rooms": rooms,
        "cities": list(cities or []),
        "city_nights": dict(city_nights_map),
        "lines": [{"name": line["name"], "total": line["total"]} for line in lines],
        "subtotal": subtotal,
        "contingency": contingency,
        "total": total,
        "budget_inr": budget_inr,
        "within": within,
        "diff": diff,
        "assump_share": assump_share,
        "rate_live": live,
        "verdict": verdict,
        "markdown": "\n".join(md),
    }