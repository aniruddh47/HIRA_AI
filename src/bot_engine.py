"""Industrial Safety assistant for TML HIRA assessments with Gemini extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from biw_reference import compact_biw_context, find_activity, find_hira_rows, load_biw_reference
except ImportError:  # Allows importing as src.bot_engine during tests.
    from .biw_reference import compact_biw_context, find_activity, find_hira_rows, load_biw_reference

try:
    from hira_standard import (
        CONTROL_HIERARCHY,
        HAZARDS,
        HIRA_STEPS,
        LEVEL_OF_HARM,
        LIKELIHOOD_SCALE,
        OVERRIDING_CRITERIA,
        PEOPLE_AFFECTED,
        REVIEW_TRIGGERS,
        RISK_LEVELS,
        SCALE_OF_RISK,
    )
except ImportError:  # Allows importing as src.bot_engine during tests.
    from .hira_standard import (
        CONTROL_HIERARCHY,
        HAZARDS,
        HIRA_STEPS,
        LEVEL_OF_HARM,
        LIKELIHOOD_SCALE,
        OVERRIDING_CRITERIA,
        PEOPLE_AFFECTED,
        REVIEW_TRIGGERS,
        RISK_LEVELS,
        SCALE_OF_RISK,
    )


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_PATH = DATA_DIR / "hira_source.txt"

if load_dotenv:
    load_dotenv()


@dataclass(frozen=True)
class BotComponents:
    gemini_model: str
    gemini_api_key: str
    gemini_api_base: str
    hira_source_available: bool


def create_components() -> BotComponents:
    """Initialize bot components used by the UI."""
    return BotComponents(
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_api_base=os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        hira_source_available=SOURCE_PATH.exists() or bool(load_biw_reference().get("available")),
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _default_state() -> dict[str, Any]:
    return {
        "activity": "",
        "hazards": [],
        "affected_people": "",
        "direct_indirect": "",
        "routine_nonroutine": "",
        "overriding_criteria": [],
        "existing_controls": "",
        "gaps": "",
        "injury_or_ill_health": "",
        "likelihood_score": None,
        "scale_score": None,
        "harm_score": None,
        "people_score": None,
        "completed": False,
        "model_used": "",
        "model_error": "",
    }


def _merge_state(state: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_state()
    if isinstance(state, dict):
        merged.update(state)
    if not isinstance(merged.get("hazards"), list):
        merged["hazards"] = []
    if not isinstance(merged.get("overriding_criteria"), list):
        merged["overriding_criteria"] = []
    return merged


def _source_excerpt() -> str:
    if not SOURCE_PATH.exists():
        return "TML HIRA source text is not available."
    text = SOURCE_PATH.read_text(encoding="utf-8", errors="replace")
    useful = []
    for marker in ("A.1      TYPE OF OHS HAZARD", "A.2        DETERMINING RISK LIKELIHOOD"):
        idx = text.find(marker)
        if idx >= 0:
            useful.append(text[idx : idx + 2600])
    return "\n\n".join(useful)[:5000] or text[:5000]


def _compact_standard_context() -> str:
    hazard_names = ", ".join(HAZARDS)
    likelihood = "; ".join(f"{score}={item['label']}" for score, item in LIKELIHOOD_SCALE.items())
    scale = "; ".join(f"{score}={item['label']}" for score, item in SCALE_OF_RISK.items())
    harm = "; ".join(f"{score}={item['label']}" for score, item in LEVEL_OF_HARM.items())
    people = "; ".join(f"{score}={label}" for score, label in PEOPLE_AFFECTED.items())
    return (
        f"Hazard types: {hazard_names}.\n"
        f"Likelihood scores: {likelihood}.\n"
        f"Scale scores: {scale}.\n"
        f"Harm scores: {harm}.\n"
        f"People scores: {people}.\n"
        "Over-riding criteria: DC=Domino Concern, LC=Legislative Concern, E=Emergency."
    )


def _gemini_extract(components: BotComponents, prompt: str) -> dict[str, Any]:
    if not components.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    model_path = quote(components.gemini_model, safe="")
    url = (
        components.gemini_api_base.rstrip("/")
        + f"/models/{model_path}:generateContent?key={components.gemini_api_key}"
    )
    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are a Tata Motors HIRA assistant. Extract facts and make conservative, safety-first "
                        "HIRA scoring judgments when the work description implies a likely risk. Return compact JSON only."
                    )
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        decoded = json.loads(response.read().decode("utf-8"))

    candidates = decoded.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise ValueError("Gemini returned empty content.")

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    parsed = json.loads(match.group(0) if match else text)
    return parsed if isinstance(parsed, dict) else {}


def _extract_with_gemini(components: BotComponents, user_text: str, state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prompt = f"""
Use the TML HIRA standard below to extract information from the latest user message.
If the message describes a work activity, infer a practical draft HIRA from it. Choose conservative
scores based on likely exposure, credible worst-case harm, people exposed, and hazard spread.

Return this JSON shape:
{{
  "activity": string or "",
  "hazards": [{{"hazard_type": one of {list(HAZARDS)}, "description": string}}], max 5 most relevant hazards only,
  "affected_people": string or "",
  "direct_indirect": "Direct" or "Indirect" or "",
  "routine_nonroutine": "Routine" or "Non-Routine" or "",
  "overriding_criteria": ["DC"|"LC"|"E"],
  "existing_controls": string or "",
  "gaps": string or "",
  "injury_or_ill_health": "Injury" or "Ill Health" or "",
  "likelihood_score": 2 or 3 or 4 or 5 or null,
  "scale_score": 2 or 3 or 4 or 5 or null,
  "harm_score": 2 or 3 or 4 or 5 or null,
  "people_score": 2 or 3 or 4 or 5 or null
}}

Current assessment state:
{json.dumps(state, ensure_ascii=False)}

TML HIRA standard guide:
{_compact_standard_context()}

BIW maintenance reference matches from A8/HIRA workbook:
{compact_biw_context(user_text)}

Latest user message:
{user_text}
"""
    try:
        return _gemini_extract(components, prompt), ""
    except (URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        return {}, str(exc)


def _is_valid_activity(text: str) -> bool:
    cleaned = _normalize_text(text)
    if not cleaned or cleaned in {"hi", "hello", "hey", "ok", "okay", "thanks", "thank you", "yes", "no"}:
        return False
    if re.fullmatch(r"[0-9]+", cleaned):
        return False
    option_words = {
        "rare",
        "occasional",
        "probable",
        "frequent",
        "direct",
        "indirect",
        "routine",
        "non routine",
        "non-routine",
        "injury",
        "ill health",
        "work area",
        "shop",
        "plant",
        "outside plant",
    }
    return cleaned not in option_words


def _infer_hazards(activity: str) -> list[dict[str, str]]:
    text = _normalize_text(activity)
    hazards: list[dict[str, str]] = []

    def add(hazard_type: str, description: str | None = None) -> None:
        if hazard_type not in [item["hazard_type"] for item in hazards]:
            hazards.append({"hazard_type": hazard_type, "description": description or HAZARDS[hazard_type]})

    if "office" in text or "desk" in text or "computer" in text:
        add("Ergonomic", "Sustained sitting, poor posture or repetitive keyboard/mouse use.")
        add("Electrical", "Sockets, adapters, chargers or extension boards may cause shock or overheating.")
        add("Fire / Explosion", "Overloaded electrical circuits or paper storage can raise fire risk.")
        add("Environmental", "Glare, poor lighting or ventilation can cause discomfort or errors.")
        add("Gravity", "Slip, trip or fall due to cables, clutter or wet floor.")

    keyword_map = {
        "Gravity": (
            "height", "ladder", "roof", "platform", "fall", "slip", "trip", "overturn", "stair", "scaffold", "edge"
        ),
        "Machinery / Tool": (
            "machine", "press", "robot", "conveyor", "tool", "fixture", "drill", "grind", "guard",
            "maintenance", "repair", "install", "alignment", "calibration", "inspection", "cutting"
        ),
        "Ergonomic": (
            "manual", "lift", "carry", "push", "pull", "repetitive", "posture", "lower", "handling", "awkward"
        ),
        "Fire / Explosion": (
            "weld", "spark", "flame", "flammable", "combustible", "gas cutting", "hot work", "short circuit", "lpg", "fuel"
        ),
        "Electrical": (
            "electric", "panel", "cable", "wire", "transformer", "live", "voltage", "socket", "switchgear", "mccb", "db"
        ),
        "Chemical": (
            "chemical", "solvent", "paint", "acid", "alkali", "fume", "toxic", "spill", "cleaning", "oil", "grease"
        ),
        "Confined Space": (
            "tank", "pit", "vessel", "manhole", "confined", "restricted access", "chamber"
        ),
        "Biological": ("waste", "sewage", "bio", "contaminated", "medical", "animal", "blood"),
        "Pressure": (
            "pressure", "hydraulic", "pneumatic", "cylinder", "compressed", "vacuum", "air line"
        ),
        "Radiation": ("xray", "x-ray", "radiation", "radiography", "laser", "uv", "infrared"),
        "Vehicular": (
            "forklift", "truck", "vehicle", "crane", "mobile equipment", "traffic", "driving", "transport", "loading"
        ),
        "Heat & Temperature": (
            "heat", "temperature", "steam", "furnace", "hot surface", "cryogenic", "burn"
        ),
        "Environmental": (
            "dust", "noise", "rain", "wind", "fog", "illumination", "vibration", "lighting", "glare", "ventilation"
        ),
        "Natural Calamity": ("earthquake", "flood", "storm", "lightning", "cyclone", "collapse"),
        "Demographic": ("public", "visitor", "crowd", "riot", "terrorism", "sabotage"),
        "Human Factors / Behavioral issues": (
            "fatigue", "rush", "night", "handover", "communication", "horseplay", "assumption", "deviation"
        ),
    }
    for hazard_type, words in keyword_map.items():
        if any(word in text for word in words):
            add(hazard_type)

    if not hazards:
        add("Ergonomic", "Awkward postures, repetitive motions or manual handling may cause strain.")
        add("Gravity", "Slip, trip or fall hazard due to walkways, tools, cables or uneven surfaces.")
        add("Human Factors / Behavioral issues", "Human behavior, assumptions, communication gaps or procedural deviation may affect safe execution.")
    return hazards


def _valid_score(value: Any) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if score in {2, 3, 4, 5} else None


def _parse_people_score(text: str) -> int | None:
    cleaned = _normalize_text(text)
    if re.search(r"\b(one|single|two)\b", cleaned):
        return 2
    if "few" in cleaned or "small team" in cleaned:
        return 3
    if "many" in cleaned:
        return 4
    match = re.search(r"\b(\d{1,4})\b", cleaned)
    if not match:
        return None
    count = int(match.group(1))
    if count <= 2:
        return 2
    if count <= 10:
        return 3
    if count <= 100:
        return 4
    return 5


def _parse_direct_score(text: str) -> int | None:
    cleaned = _normalize_text(text)
    if re.fullmatch(r"[2-5]", cleaned):
        return int(cleaned)
    match = re.search(r"\bscore\s*[:=-]?\s*([2-5])\b", cleaned)
    return int(match.group(1)) if match else None


def _parse_likelihood_score(text: str) -> int | None:
    cleaned = _normalize_text(text)
    if any(word in cleaned for word in ("rare", "very low")):
        return 2
    if any(word in cleaned for word in ("occasional", "once in 3 years", "once in three years")):
        return 3
    if any(word in cleaned for word in ("probable", "once in a year", "yearly", "annual")):
        return 4
    if any(word in cleaned for word in ("frequent", "several times", "daily", "weekly", "monthly")):
        return 5
    return None


def _parse_scale_score(text: str) -> int | None:
    cleaned = _normalize_text(text)
    if "outside plant" in cleaned or "external population" in cleaned:
        return 5
    if "plant boundary" in cleaned or re.search(r"\bplant\b", cleaned):
        return 4
    if "shop" in cleaned or "shed" in cleaned or "adjoining" in cleaned:
        return 3
    if "work area" in cleaned or "machine area" in cleaned or "welding area" in cleaned or "grinding area" in cleaned:
        return 2
    return None


def _parse_harm_score(text: str) -> int | None:
    cleaned = _normalize_text(text)
    if any(word in cleaned for word in ("fatal", "death", "permanent disability", "chronic", "notifiable")):
        return 5
    if any(word in cleaned for word in ("lost time", "restricted work", "medical treatment", "hipo", "serious")):
        return 4
    if any(word in cleaned for word in ("first aid", "minor health", "moderate damage", "harmful")):
        return 3
    if any(word in cleaned for word in ("insignificant", "discomfort", "minor near miss", "no damage")):
        return 2
    return None


def _parse_overriding(text: str) -> list[str]:
    cleaned = _normalize_text(text)
    if cleaned in {"none", "no", "nil", "na", "n/a", "no overriding", "no over-riding", "not applicable"}:
        return ["None"]
    found = []
    if "domino" in cleaned or re.search(r"\bdc\b", cleaned):
        found.append("DC")
    if "legal" in cleaned or "legislative" in cleaned or re.search(r"\blc\b", cleaned):
        found.append("LC")
    if "emergency" in cleaned or re.search(r"\be\b", cleaned):
        found.append("E")
    return found


def _apply_extracted(state: dict[str, Any], extracted: dict[str, Any]) -> None:
    activity = str(extracted.get("activity") or "").strip()
    if activity and not state["activity"] and _is_valid_activity(activity):
        state["activity"] = activity

    hazards = extracted.get("hazards")
    if isinstance(hazards, list):
        clean_hazards = []
        for item in hazards:
            if not isinstance(item, dict):
                continue
            hazard_type = str(item.get("hazard_type") or "").strip()
            if hazard_type in HAZARDS:
                description = str(item.get("description") or HAZARDS[hazard_type]).strip()
                clean_hazards.append({"hazard_type": hazard_type, "description": description})
        if clean_hazards and len(clean_hazards) <= 5 and not state["hazards"]:
            state["hazards"] = clean_hazards

    for key in ("affected_people", "direct_indirect", "routine_nonroutine", "existing_controls", "gaps", "injury_or_ill_health"):
        value = str(extracted.get(key) or "").strip()
        if value and not state.get(key):
            state[key] = value

    criteria = extracted.get("overriding_criteria")
    if isinstance(criteria, list):
        current = set(state["overriding_criteria"])
        for item in criteria:
            code = str(item).strip().upper()
            if code in OVERRIDING_CRITERIA:
                current.add(code)
            elif code == "NONE":
                current.add("None")
        state["overriding_criteria"] = sorted(current)

    for key in ("likelihood_score", "scale_score", "harm_score", "people_score"):
        score = _valid_score(extracted.get(key))
        if score is not None and state.get(key) is None:
            state[key] = score


def _rule_based_extract(user_text: str, state: dict[str, Any]) -> dict[str, Any]:
    text = user_text.strip()
    cleaned = _normalize_text(text)
    extracted: dict[str, Any] = {}

    if not state["activity"]:
        if _is_valid_activity(text):
            extracted["activity"] = text
            hazards = _infer_hazards(text)
            if hazards:
                extracted["hazards"] = hazards[:5]
        return extracted

    if not state["affected_people"]:
        extracted["affected_people"] = text
        score = _parse_people_score(text)
        if score is not None:
            extracted["people_score"] = score
        return extracted

    if not state["direct_indirect"]:
        if "direct" in cleaned and "indirect" not in cleaned:
            extracted["direct_indirect"] = "Direct"
        elif "indirect" in cleaned:
            extracted["direct_indirect"] = "Indirect"
        return extracted

    if not state["routine_nonroutine"]:
        if "non-routine" in cleaned or "non routine" in cleaned:
            extracted["routine_nonroutine"] = "Non-Routine"
        elif "routine" in cleaned:
            extracted["routine_nonroutine"] = "Routine"
        return extracted

    if not state["overriding_criteria"]:
        criteria = _parse_overriding(text)
        if criteria:
            extracted["overriding_criteria"] = criteria
        return extracted

    if not state["existing_controls"]:
        if text:
            extracted["existing_controls"] = text
        return extracted

    if not state["gaps"]:
        if text:
            extracted["gaps"] = text
        return extracted

    if not state["injury_or_ill_health"]:
        if "ill health" in cleaned:
            extracted["injury_or_ill_health"] = "Ill Health"
        elif "injury" in cleaned:
            extracted["injury_or_ill_health"] = "Injury"
        return extracted

    if state["likelihood_score"] is None:
        score = _parse_likelihood_score(text)
        if score is not None:
            extracted["likelihood_score"] = score
        return extracted

    if state["scale_score"] is None:
        score = _parse_scale_score(text)
        if score is not None:
            extracted["scale_score"] = score
        return extracted

    if state["harm_score"] is None:
        score = _parse_harm_score(text)
        if score is not None:
            extracted["harm_score"] = score
        return extracted

    if state["people_score"] is None:
        score = _parse_people_score(text)
        if score is not None:
            extracted["people_score"] = score
        return extracted

    return extracted


def _activity_text(state: dict[str, Any]) -> str:
    parts = [
        state.get("activity", ""),
        state.get("affected_people", ""),
        state.get("existing_controls", ""),
        state.get("gaps", ""),
    ]
    hazard_text = " ".join(
        f"{item.get('hazard_type', '')} {item.get('description', '')}"
        for item in state.get("hazards", [])
        if isinstance(item, dict)
    )
    return _normalize_text(" ".join(parts + [hazard_text]))


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _infer_direct_indirect(text: str) -> str:
    if _has_any(text, ("contractor", "vendor", "visitor", "third party", "agency")):
        return "Indirect"
    return "Direct"


def _strong_nonroutine_text(text: str) -> bool:
    return _has_any(
        text,
        (
            "change",
            "changing",
            "replace",
            "replacement",
            "repair",
            "modify",
            "install",
            "remove",
            "shutdown",
            "breakdown",
            "new task",
            "not in sop",
            "no sop",
            "roof",
            "rooftop",
            "height",
        ),
    )


def _infer_routine_nonroutine(text: str) -> str:
    activity = find_activity(text)
    if activity and activity.routine:
        return activity.routine

    return "Non-Routine" if _strong_nonroutine_text(text) else "Routine"


def _clean_hazard_type(value: str) -> str:
    normalized = _normalize_text(value)
    for hazard_type in HAZARDS:
        if _normalize_text(hazard_type) == normalized:
            return hazard_type
    if normalized == "chemical":
        return "Chemical"
    return value.strip()


def _criteria_from_reference(value: str) -> list[str]:
    text = _normalize_text(value)
    found = []
    if "dc" in text or "domino" in text:
        found.append("DC")
    if "lc" in text or "legal" in text or "legislative" in text:
        found.append("LC")
    if re.search(r"\be\b", text) or "emergency" in text:
        found.append("E")
    return found


def _max_score(values: list[int | None]) -> int | None:
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _merge_unique(values: list[str], fallback: str = "") -> str:
    seen: set[str] = set()
    output = []
    for value in values:
        clean = re.sub(r"\s+", " ", str(value).strip())
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            output.append(clean)
    return "; ".join(output) if output else fallback


def _apply_biw_reference(state: dict[str, Any]) -> None:
    activity_text = state.get("activity", "")
    if not activity_text:
        return

    rows = find_hira_rows(activity_text, limit=8)
    activity = find_activity(activity_text)

    existing_hazards = list(state.get("hazards", []))

    generic_only = (
        len(state["hazards"]) == 1
        and state["hazards"][0].get("hazard_type") == "Human Factors / Behavioral issues"
    )
    if rows:
        hazards = []
        seen = set()
        for row in rows:
            hazard_type = _clean_hazard_type(row.hazard_type)
            if hazard_type not in HAZARDS:
                continue
            key = (hazard_type, row.description)
            if key in seen:
                continue
            seen.add(key)
            hazards.append({"hazard_type": hazard_type, "description": row.description or HAZARDS[hazard_type]})
            if len(hazards) >= 5:
                break
        if hazards:
            merged = list(hazards)
            for item in existing_hazards:
                if len(merged) >= 5:
                    break
                hazard_type = item.get("hazard_type")
                description = item.get("description")
                key = (hazard_type, description)
                if key in {(haz.get("hazard_type"), haz.get("description")) for haz in merged}:
                    continue
                merged.append(item)
            state["hazards"] = merged

    if rows:
        direct = next((row.direct_indirect for row in rows if row.direct_indirect), "")
        if direct:
            state["direct_indirect"] = direct
    if not state["direct_indirect"]:
        direct = next((row.direct_indirect for row in rows if row.direct_indirect), "")
        state["direct_indirect"] = direct or _infer_direct_indirect(_activity_text(state))

    routine = activity.routine if activity and activity.routine else next((row.routine_nonroutine for row in rows if row.routine_nonroutine), "")
    if routine:
        state["routine_nonroutine"] = routine
    elif not state["routine_nonroutine"]:
        state["routine_nonroutine"] = _infer_routine_nonroutine(activity_text)
    elif not activity and _strong_nonroutine_text(activity_text):
        state["routine_nonroutine"] = "Non-Routine"

    if rows:
        criteria = []
        if activity and activity.emergency:
            criteria.append("E")
        for row in rows:
            criteria.extend(_criteria_from_reference(row.overriding_criteria))
        state["overriding_criteria"] = sorted(set(criteria)) if criteria else ["None"]
    elif not state["overriding_criteria"]:
        criteria: list[str] = []
        if activity and activity.emergency:
            criteria.append("E")
        for row in rows:
            criteria.extend(_criteria_from_reference(row.overriding_criteria))
        state["overriding_criteria"] = sorted(set(criteria)) if criteria else _infer_overriding_criteria(_activity_text(state), state["hazards"])

    if rows:
        state["existing_controls"] = _merge_unique(
            [row.existing_controls for row in rows],
            fallback=state.get("existing_controls", ""),
        )

    if rows:
        gaps = _merge_unique([row.gaps for row in rows], fallback=state.get("gaps", ""))
        state["gaps"] = gaps

    if rows:
        outcome = next((row.outcome for row in rows if row.outcome), "")
        if outcome:
            state["injury_or_ill_health"] = "Ill Health" if "ill" in _normalize_text(outcome) else "Injury"
    elif not state["injury_or_ill_health"]:
        outcome = next((row.outcome for row in rows if row.outcome), "")
        if outcome:
            state["injury_or_ill_health"] = "Ill Health" if "ill" in _normalize_text(outcome) else "Injury"

    if state["likelihood_score"] is None:
        state["likelihood_score"] = _max_score([row.likelihood_score for row in rows])
    if state["scale_score"] is None:
        state["scale_score"] = _max_score([row.scale_score for row in rows])
    if state["harm_score"] is None:
        state["harm_score"] = _max_score([row.harm_score for row in rows])


def _infer_overriding_criteria(text: str, hazards: list[dict[str, str]]) -> list[str]:
    hazard_types = {item.get("hazard_type") for item in hazards if isinstance(item, dict)}
    criteria: list[str] = []
    if _has_any(text, ("emergency", "rescue", "evacuation", "collapse", "explosion")):
        criteria.append("E")
    if hazard_types.intersection({"Electrical", "Fire / Explosion", "Confined Space", "Pressure"}):
        criteria.append("LC")
    if "Gravity" in hazard_types and _has_any(text, ("height", "roof", "rooftop", "ladder", "scaffold")):
        criteria.append("LC")
    return criteria or ["None"]


def _infer_likelihood_score(text: str, hazards: list[dict[str, str]]) -> int:
    parsed = _parse_likelihood_score(text)
    if parsed is not None:
        return parsed
    hazard_types = {item.get("hazard_type") for item in hazards if isinstance(item, dict)}
    if _has_any(text, ("daily", "weekly", "continuous", "frequent")):
        return 5
    if hazard_types.intersection({"Gravity", "Electrical", "Fire / Explosion", "Confined Space", "Vehicular", "Pressure"}):
        return 4
    if hazard_types.intersection({"Machinery / Tool", "Chemical", "Heat & Temperature", "Environmental"}):
        return 3
    return 3


def _infer_scale_score(text: str) -> int:
    parsed = _parse_scale_score(text)
    if parsed is not None:
        return parsed
    if _has_any(text, ("public", "outside plant", "external", "nearby community")):
        return 5
    if _has_any(text, ("plant", "multiple shops", "factory")):
        return 4
    if _has_any(text, ("warehouse", "shop", "shed", "roof", "rooftop", "adjoining")):
        return 3
    return 2


def _infer_harm_score(text: str, hazards: list[dict[str, str]]) -> int:
    parsed = _parse_harm_score(text)
    if parsed is not None:
        return parsed
    hazard_types = {item.get("hazard_type") for item in hazards if isinstance(item, dict)}
    if hazard_types.intersection({"Gravity", "Electrical", "Fire / Explosion", "Confined Space", "Pressure"}):
        return 5
    if hazard_types.intersection({"Vehicular", "Machinery / Tool", "Chemical", "Heat & Temperature"}):
        return 4
    if hazard_types.intersection({"Ergonomic", "Environmental", "Biological", "Radiation"}):
        return 3
    return 4


def _infer_people_score_from_state(state: dict[str, Any], text: str) -> int:
    parsed = _parse_people_score(state.get("affected_people", "") or text)
    if parsed is not None:
        return parsed
    if _has_any(text, ("public", "visitors", "many", "crowd", "full shift")):
        return 4
    if _has_any(text, ("one", "single", "alone")):
        return 2
    return 3


def _infer_outcome_type(text: str, hazards: list[dict[str, str]]) -> str:
    hazard_types = {item.get("hazard_type") for item in hazards if isinstance(item, dict)}
    if hazard_types.intersection({"Chemical", "Biological", "Radiation", "Environmental", "Ergonomic"}):
        return "Ill Health"
    return "Injury"


def _auto_complete_assessment(state: dict[str, Any]) -> None:
    if not state.get("activity"):
        return

    _apply_biw_reference(state)

    if not state["hazards"]:
        state["hazards"] = _infer_hazards(state["activity"])[:5]

    text = _activity_text(state)
    if not state["direct_indirect"]:
        state["direct_indirect"] = _infer_direct_indirect(text)
    if not state["routine_nonroutine"]:
        state["routine_nonroutine"] = _infer_routine_nonroutine(text)
    elif not find_activity(state["activity"]) and _strong_nonroutine_text(state["activity"]):
        state["routine_nonroutine"] = "Non-Routine"
    if not state["overriding_criteria"]:
        state["overriding_criteria"] = _infer_overriding_criteria(text, state["hazards"])
    if not state["existing_controls"]:
        state["existing_controls"] = "Permit-to-work, area barricading, competent supervision, SOP/JSA briefing and task-specific PPE to be verified before work"
    if not state["gaps"]:
        state["gaps"] = "Verify isolation, access, emergency readiness, site conditions and authorization before starting"
    if not state["injury_or_ill_health"]:
        state["injury_or_ill_health"] = _infer_outcome_type(text, state["hazards"])
    if state["likelihood_score"] is None:
        state["likelihood_score"] = _infer_likelihood_score(text, state["hazards"])
    if state["scale_score"] is None:
        state["scale_score"] = _infer_scale_score(text)
    if state["harm_score"] is None:
        state["harm_score"] = _infer_harm_score(text, state["hazards"])
    if state["affected_people"] and state["people_score"] is None:
        state["people_score"] = _infer_people_score_from_state(state, text)


def _update_state_from_user_input(state: dict[str, Any], user_text: str, components: BotComponents | None) -> None:
    text = user_text.strip()
    if not text:
        return

    if components is None:
        state["model_used"] = "deterministic"
        state["model_error"] = "Gemini components are not configured."
        extracted = _rule_based_extract(text, state)
        if extracted:
            _apply_extracted(state, extracted)
        return

    extracted, error = _extract_with_gemini(components, text, state)
    if extracted:
        state["model_used"] = "gemini"
        state["model_error"] = ""
        _apply_extracted(state, extracted)
    else:
        fallback = _rule_based_extract(text, state)
        if fallback:
            state["model_used"] = "deterministic"
            state["model_error"] = error or "Gemini unavailable; using deterministic fallback."
            _apply_extracted(state, fallback)
        else:
            state["model_used"] = "gemini"
            state["model_error"] = error or "No structured extraction returned by Gemini."


def _next_question(state: dict[str, Any]) -> str | None:
    if not state["activity"]:
        return "Describe the process/activity/sub-activity for HIRA."
    if not state["affected_people"]:
        return "How many people may be involved or exposed during this task? Mention employees, contractors, visitors, nearby operators, or public."
    if not state["direct_indirect"]:
        return "Is this a Direct Tata Motors activity or Indirect contractor/visitor/vendor activity?"
    if not state["routine_nonroutine"]:
        return "Is this Routine or Non-Routine? If non-routine, mention whether SOP/SMP/WIS is unavailable or changed."
    if not state["overriding_criteria"]:
        return "Any over-riding criteria: Domino Concern (DC), Legislative Concern (LC), Emergency (E), or none?"
    if not state["existing_controls"]:
        return "What existing controls are already available? Example: LOTO, machine guard, permit, SOP, fire extinguisher, ventilation, PPE."
    if not state["gaps"]:
        return "What gaps remain between hazards and controls? Say 'no major gap' if none."
    if not state["injury_or_ill_health"]:
        return "Is the primary outcome Injury or Ill Health?"
    if state["likelihood_score"] is None:
        return "Select likelihood: Rare=2, Occasional/once in 3 years=3, Probable/once in a year=4, Frequent/several times in a year=5."
    if state["scale_score"] is None:
        return "Select scale of risk: Work area=2, Shop area=3, Plant boundary=4, Outside plant boundary=5."
    if state["harm_score"] is None:
        return "Select level of harm: Insignificant=2, Harmful/first aid=3, Very harmful/LTI/HIPO=4, Extremely harmful/fatality=5."
    if state["people_score"] is None:
        return "Select number of people affected: 0-2=2, 3-10=3, 11-100=4, 101 and above=5."
    return None


def _risk_level(rpn: int, overriding_criteria: list[str]) -> dict[str, Any]:
    for level in RISK_LEVELS:
        if int(level["min"]) <= rpn <= int(level["max"]):
            result = dict(level)
            break
    else:
        result = dict(RISK_LEVELS[-1])

    active_overrides = [code for code in overriding_criteria if code in OVERRIDING_CRITERIA]
    if active_overrides and result["acceptability"] == "Acceptable Risk":
        result = dict(next(item for item in RISK_LEVELS if item["level"] == "Moderate"))
        result["action"] = "Over-riding criteria apply, so the risk must be treated as significant/unacceptable and controlled before work."
    return result


def _controls_for_hazard(hazard_type: str, state: dict[str, Any]) -> dict[str, str]:
    base = {
        "Elimination": "Remove the direct source of exposure or avoid the activity where practical.",
        "Substitution": "Use a lower-risk method, material, equipment or work sequence.",
        "Engineering controls": "Provide physical safeguards, barriers, isolation, ventilation, guards or interlocks.",
        "Administrative controls": "Use permit, SOP/SMP/WIS, Take-2/JSA, training, supervision, signage and restricted access.",
        "PPE": "Use task-specific PPE such as helmet, eye protection, gloves, safety shoes, hearing or respiratory protection.",
    }
    gravity_controls = {
        "Elimination": "Keep people and loose tools away from edges, openings and suspended/drop zones where practical.",
        "Substitution": "Use a safer access method, tool holder or mechanical aid to reduce slip, trip, fall and dropped-object exposure.",
        "Engineering controls": "Provide firm access, toe boards, tool lanyards, covers, guardrails, barricades and good illumination where required.",
        "Administrative controls": "Use area inspection, housekeeping, exclusion zone, competent supervision and pre-job briefing.",
        "PPE": "Use helmet, safety shoes, gloves and task-specific fall protection where height exposure exists.",
    }
    if _has_any(_activity_text(state), ("roof", "rooftop", "height", "ladder", "scaffold", "fall from height")):
        gravity_controls = {
            "Elimination": "Avoid roof or height access where the job can be done from ground level or by pre-fabrication.",
            "Substitution": "Use safer access equipment such as a certified scaffold or mobile elevated work platform instead of ladders where feasible.",
            "Engineering controls": "Provide certified scaffolding, guardrails, lifelines, anchor points, roof edge protection, covers for fragile sheets and safe access routes.",
            "Administrative controls": "Use work-at-height permit, rescue plan, weather check, exclusion zone, competent supervision and pre-job briefing.",
            "PPE": "Use full-body harness with double lanyard or fall arrester, helmet with chin strap, safety shoes and cut-resistant gloves.",
        }
    specific = {
        "Gravity": gravity_controls,
        "Electrical": {
            "Elimination": "De-energize and isolate affected circuits before contact.",
            "Engineering controls": "Apply LOTO, insulated barriers, covered terminals and verified grounding.",
            "Administrative controls": "Use electrical permit, authorized person and test-before-touch verification.",
            "PPE": "Use arc-rated face shield, insulated gloves, helmet and dielectric safety shoes.",
        },
        "Fire / Explosion": {
            "Elimination": "Remove combustibles and isolate ignition sources.",
            "Substitution": "Use cold-work method where feasible.",
            "Engineering controls": "Use spark containment, gas checks, ventilation and fire extinguishers.",
            "Administrative controls": "Use hot-work permit, fire watch and post-work fire monitoring.",
            "PPE": "Use flame-resistant clothing, face protection, heat-resistant gloves and safety shoes.",
        },
        "Confined Space": {
            "Elimination": "Avoid entry using external cleaning or inspection methods where possible.",
            "Engineering controls": "Provide ventilation, gas monitoring, entry barriers and rescue retrieval setup.",
            "Administrative controls": "Use confined-space permit, standby watch, communication checks and rescue readiness.",
            "PPE": "Use respiratory protection where required, helmet, gloves, harness and safety shoes.",
        },
        "Machinery / Tool": {
            "Elimination": "Stop and isolate machine motion before intervention.",
            "Engineering controls": "Use guards, interlocks, emergency stops and mechanical blocking.",
            "Administrative controls": "Use task SOP, LOTO checklist and controlled restart authorization.",
            "PPE": "Use cut-resistant gloves, eye protection, helmet and safety shoes.",
        },
        "Vehicular": {
            "Engineering controls": "Separate pedestrians and vehicles using barricades, marked lanes and alarms.",
            "Administrative controls": "Use traffic management plan, speed limits, banksman and authorization.",
            "PPE": "Use high-visibility jacket, helmet and safety shoes.",
        },
    }
    base.update(specific.get(hazard_type, {}))
    if state.get("routine_nonroutine") == "Non-Routine":
        base["Administrative controls"] += " Non-routine work should also refer Take-2, JSA and permit-to-work standards."
    return base


def _md_cell(value: Any) -> str:
    """Format a value safely for a Markdown table cell."""
    text = str(value).strip()
    text = text.replace("|", "\\|")
    text = re.sub(r"\s+", " ", text)
    return text or "-"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_text(text)))


def _split_human_factors(description: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", description.strip())
    if "may affect safe execution" not in cleaned.lower():
        return [cleaned] if cleaned else []
    base = re.sub(r"\bmay affect safe execution\.?\b", "", cleaned, flags=re.IGNORECASE).strip()
    parts = [part.strip() for part in re.split(r",|\bor\b|/|;|\band\b", base) if part.strip()]
    if not parts:
        return [cleaned]
    return [f"{part} may affect safe execution." for part in parts]


def _expanded_hazards(state: dict[str, Any]) -> list[dict[str, str]]:
    hazards = []
    for item in state.get("hazards", []):
        hazard_type = str(item.get("hazard_type") or "").strip()
        description = str(item.get("description") or "").strip()
        if not hazard_type:
            continue
        if hazard_type == "Human Factors / Behavioral issues" and description:
            for desc in _split_human_factors(description):
                hazards.append({"hazard_type": hazard_type, "description": desc})
            continue
        hazards.append({"hazard_type": hazard_type, "description": description})
    return hazards


def _best_reference_row(hazard: dict[str, str], rows: list[HiraReferenceRow]) -> HiraReferenceRow | None:
    hazard_type = _normalize_text(hazard.get("hazard_type", ""))
    hazard_desc = _normalize_text(hazard.get("description", ""))
    if not hazard_type:
        return None
    best = None
    best_score = 0
    hazard_tokens = _tokenize(hazard_desc)
    for row in rows:
        row_type = _normalize_text(_clean_hazard_type(row.hazard_type))
        if row_type != hazard_type:
            continue
        row_desc = _normalize_text(row.description)
        score = 5
        if hazard_desc and (hazard_desc in row_desc or row_desc in hazard_desc):
            score += 8
        score += len(hazard_tokens.intersection(_tokenize(row_desc)))
        if score > best_score:
            best_score = score
            best = row
    return best


def _hazard_assessments(state: dict[str, Any]) -> list[dict[str, Any]]:
    activity_text = state.get("activity", "")
    rows = find_hira_rows(activity_text, limit=8) if activity_text else []
    hazards = _expanded_hazards(state)
    assessments = []
    override = [code for code in state.get("overriding_criteria", []) if code in OVERRIDING_CRITERIA]

    for hazard in hazards:
        reference = _best_reference_row(hazard, rows) if rows else None
        likelihood = reference.likelihood_score if reference else state.get("likelihood_score")
        scale = reference.scale_score if reference else state.get("scale_score")
        harm = reference.harm_score if reference else state.get("harm_score")
        people = reference.people_score if reference else state.get("people_score")
        if None in (likelihood, scale, harm, people):
            continue
        severity = int(scale) * int(harm) * int(people)
        rpn = int(likelihood) * severity
        risk = _risk_level(rpn, override)
        assessments.append(
            {
                "activity": state.get("activity", ""),
                "hazard_type": hazard.get("hazard_type", ""),
                "description": hazard.get("description", ""),
                "affected_people": reference.affected_people if reference and reference.affected_people else state.get("affected_people", ""),
                "outcome": reference.outcome if reference and reference.outcome else state.get("injury_or_ill_health", ""),
                "direct_indirect": reference.direct_indirect if reference and reference.direct_indirect else state.get("direct_indirect", ""),
                "routine_nonroutine": reference.routine_nonroutine if reference and reference.routine_nonroutine else state.get("routine_nonroutine", ""),
                "overriding_criteria": override,
                "existing_controls": reference.existing_controls if reference and reference.existing_controls else state.get("existing_controls", ""),
                "gaps": reference.gaps if reference and reference.gaps else state.get("gaps", ""),
                "likelihood": int(likelihood),
                "scale": int(scale),
                "harm": int(harm),
                "people": int(people),
                "severity": severity,
                "rpn": rpn,
                "risk_level": risk["level"],
                "acceptability": risk["acceptability"],
                "required_action": risk["action"],
            }
        )

    return assessments


def build_hazard_assessments(state: dict[str, Any]) -> list[dict[str, Any]]:
    return _hazard_assessments(state)


def _build_final_summary(state: dict[str, Any]) -> str:
    likelihood = int(state["likelihood_score"])
    scale = int(state["scale_score"])
    harm = int(state["harm_score"])
    people = int(state["people_score"])
    severity = scale * harm * people
    rpn = likelihood * severity
    risk = _risk_level(rpn, state["overriding_criteria"])

    detail_rows = [
        ("Activity", state["activity"]),
        ("Affected people", f"{state['affected_people']} ({PEOPLE_AFFECTED[people]}, Score {people})"),
        ("Direct / Indirect", state["direct_indirect"]),
        ("Routine / Non-Routine", state["routine_nonroutine"]),
        ("Outcome type", state["injury_or_ill_health"]),
        ("Existing controls", state["existing_controls"]),
        ("Remaining gaps", state["gaps"]),
    ]
    detail_table = "\n".join(
        ["| Field | Details |", "|---|---|"]
        + [f"| {_md_cell(label)} | {_md_cell(value)} |" for label, value in detail_rows]
    )

    hazards = _expanded_hazards(state)
    hazard_table = "\n".join(
        ["| Hazard type | Description |", "|---|---|"]
        + [
            f"| {_md_cell(item['hazard_type'])} | {_md_cell(item['description'])} |"
            for item in hazards
        ]
    )

    overriding = [code for code in state["overriding_criteria"] if code != "None"] or ["None"]
    overriding_rows = [
        (code, OVERRIDING_CRITERIA[code]) for code in overriding if code in OVERRIDING_CRITERIA
    ] or [("None", "No over-riding criteria identified")]
    overriding_table = "\n".join(
        ["| Criteria | Meaning |", "|---|---|"]
        + [f"| {_md_cell(code)} | {_md_cell(description)} |" for code, description in overriding_rows]
    )

    per_hazard_rows = []
    for item in _hazard_assessments(state):
        severity_label = f"{item['scale']} x {item['harm']} x {item['people']}"
        rpn_label = f"{item['likelihood']} x {item['severity']}"
        per_hazard_rows.append(
            (
                item["hazard_type"],
                item["description"],
                severity_label,
                rpn_label,
                item["rpn"],
                f"{item['risk_level']} ({item['acceptability']})",
            )
        )
    per_hazard_table = "\n".join(
        ["| Hazard type | Description | Severity | RPN formula | RPN | Risk level |", "|---|---|---:|---:|---:|---|"]
        + [
            f"| {_md_cell(hazard_type)} | {_md_cell(description)} | {_md_cell(severity)} | {_md_cell(rpn_formula)} | {_md_cell(rpn)} | {_md_cell(risk_level)} |"
            for hazard_type, description, severity, rpn_formula, rpn, risk_level in per_hazard_rows
        ]
    )

    risk_rows = [
        ("Likelihood", LIKELIHOOD_SCALE[likelihood]["label"], likelihood),
        ("Scale of risk", SCALE_OF_RISK[scale]["label"], scale),
        ("Level of harm", LEVEL_OF_HARM[harm]["label"], harm),
        ("People affected", PEOPLE_AFFECTED[people], people),
        ("Severity", f"Scale x Harm x People = {scale} x {harm} x {people}", severity),
        ("RPN", f"Likelihood x Severity = {likelihood} x {severity}", rpn),
        ("Risk level", f"{risk['level']} ({risk['acceptability']})", ""),
        ("Required action", risk["action"], ""),
    ]
    risk_table = "\n".join(
        ["| Item | Details | Score / Value |", "|---|---|---:|"]
        + [
            f"| {_md_cell(item)} | {_md_cell(details)} | {_md_cell(score)} |"
            for item, details, score in risk_rows
        ]
    )

    control_rows = []
    for control_type in CONTROL_HIERARCHY:
        seen_controls = set()
        for item in state["hazards"]:
            hazard_type = item["hazard_type"]
            control = _controls_for_hazard(hazard_type, state)[control_type]
            key = (control_type, hazard_type, control)
            if key not in seen_controls:
                control_rows.append((control_type, hazard_type, control))
                seen_controls.add(key)
    control_table = "\n".join(
        ["| Control type | Hazard type | Recommended measure |", "|---|---|---|"]
        + [
            f"| {_md_cell(control_type)} | {_md_cell(hazard_type)} | {_md_cell(control)} |"
            for control_type, hazard_type, control in control_rows
        ]
    )

    process_table = "\n".join(
        ["| Step | HIRA process |", "|---:|---|"]
        + [f"| {index} | {_md_cell(step)} |" for index, step in enumerate(HIRA_STEPS, start=1)]
    )

    review_table = "\n".join(
        ["| Step | Review trigger |", "|---:|---|"]
        + [f"| {index} | {_md_cell(item)} |" for index, item in enumerate(REVIEW_TRIGGERS[:6], start=1)]
    )
    status_table = "\n".join(
        [
            "| Status | Note |",
            "|---|---|",
            "| Draft for review | Do not treat as final approval to start work. Review by line manager / shop manager / EHS is required. |",
        ]
    )

    return (
        "### TML HIRA Draft\n"
        f"{detail_table}\n\n"
        "### Hazards\n"
        f"{hazard_table}\n\n"
        "### Over-Riding Criteria\n"
        f"{overriding_table}\n\n"
        "### Risk Calculation (Per Hazard)\n"
        f"{per_hazard_table}\n\n"
        "### Risk Calculation (Overall)\n"
        f"{risk_table}\n\n"
        "### Control Measures\n"
        f"{control_table}\n\n"
        "### HIRA Process Reminder\n"
        f"{process_table}\n\n"
        "### Review Triggers\n"
        f"{review_table}\n\n"
        "### Review Status\n"
        f"{status_table}"
    )


def assess_question(
    llm: BotComponents | Any,
    user_question: str,
    additional_details: str = "",
    conversation_history: str = "",
) -> dict[str, Any]:
    """Run a TML HIRA conversation with Gemini extraction and deterministic scoring."""
    del conversation_history

    components = llm if isinstance(llm, BotComponents) else create_components()
    prior_state: dict[str, Any] | None = None
    if additional_details:
        try:
            decoded = json.loads(additional_details)
            if isinstance(decoded, dict):
                prior_state = decoded
        except json.JSONDecodeError:
            prior_state = None

    state = _merge_state(prior_state)
    if state.get("completed"):
        fresh_state = _default_state()
        _update_state_from_user_input(fresh_state, user_question, components)
        if _is_valid_activity(fresh_state.get("activity", "")):
            state = fresh_state
        else:
            return {
                "answer": "Share the next process/activity/sub-activity when you want to start another TML HIRA draft.",
                "found_docs": SOURCE_PATH.exists() or bool(load_biw_reference().get("available")),
                "state": state,
                "completed": True,
            }
    else:
        _update_state_from_user_input(state, user_question, components)

    _auto_complete_assessment(state)
    question = _next_question(state)
    if question:
        return {
            "answer": question,
            "found_docs": SOURCE_PATH.exists() or bool(load_biw_reference().get("available")),
            "state": state,
            "completed": False,
        }

    state["completed"] = True
    return {
        "answer": _build_final_summary(state),
        "found_docs": SOURCE_PATH.exists() or bool(load_biw_reference().get("available")),
        "state": state,
        "completed": True,
    }


def main() -> None:
    """Run an interactive terminal HIRA conversation loop."""
    llm = create_components()
    state = _default_state()

    print("TML HIRA Assistant")
    print("Type 'exit' to quit.")
    print("Describe the process/activity/sub-activity for HIRA.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting HIRA assistant.")
            break

        result = assess_question(llm=llm, user_question=user_input, additional_details=json.dumps(state))
        state = result.get("state", state)
        print(f"Assistant: {result['answer']}")


if __name__ == "__main__":
    main()
