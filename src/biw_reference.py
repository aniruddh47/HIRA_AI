"""BIW maintenance reference reader for the HIRA assistant.

The project avoids an Excel dependency by reading the workbook XML directly.
Only the fields needed by the chatbot are extracted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BIW_WORKBOOK_PATH = DATA_DIR / "BIW_MAINT.xlsx"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OD_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"m": MAIN_NS, "rel": REL_NS}


@dataclass(frozen=True)
class BiwActivity:
    name: str
    area: str
    location: str
    routine: str
    emergency: bool


@dataclass(frozen=True)
class HiraReferenceRow:
    activity: str
    hazard_type: str
    description: str
    affected_people: str
    outcome: str
    direct_indirect: str
    routine_nonroutine: str
    overriding_criteria: str
    existing_controls: str
    gaps: str
    likelihood_score: int | None
    scale_score: int | None
    harm_score: int | None
    people_score: int | None
    rpn: int | None
    risk_level: str


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _col_number(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    number = 0
    for char in letters:
        number = number * 26 + ord(char.upper()) - 64
    return number


def _shared_strings(book: ZipFile) -> list[str]:
    root = ET.fromstring(book.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("m:si", NS):
        values.append("".join(text.text or "" for text in item.findall(".//m:t", NS)))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("m:v", NS)
    if cell_type == "s" and value is not None and value.text is not None:
        return shared[int(value.text)]
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//m:t", NS))
    return value.text if value is not None and value.text is not None else ""


def _sheet_paths(book: ZipFile) -> dict[str, str]:
    rel_root = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    rels = {
        rel.attrib["Id"]: "xl/" + rel.attrib["Target"]
        for rel in rel_root.findall("rel:Relationship", NS)
    }
    workbook = ET.fromstring(book.read("xl/workbook.xml"))
    paths = {}
    for sheet in workbook.findall("m:sheets/m:sheet", {"m": MAIN_NS}):
        rel_id = sheet.attrib[f"{{{OD_REL_NS}}}id"]
        paths[sheet.attrib["name"]] = rels[rel_id]
    return paths


def _rows(book: ZipFile, sheet_path: str) -> list[list[str]]:
    shared = _shared_strings(book)
    root = ET.fromstring(book.read(sheet_path))
    output: list[list[str]] = []
    for row in root.findall(".//m:sheetData/m:row", NS):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", NS):
            cells[_col_number(cell.attrib["r"])] = _cell_value(cell, shared).strip()
        if any(cells.values()):
            last_col = max(cells) if cells else 0
            output.append([cells.get(index, "") for index in range(1, last_col + 1)])
    return output


def _score(value: str) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed in {2, 3, 4, 5} else None


def _rpn(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _routine_from_flags(routine_flag: str, nonroutine_flag: str, emergency_flag: str) -> tuple[str, bool]:
    first_flag = _normalize(routine_flag)
    if first_flag == "nr":
        return "Non-Routine", _normalize(emergency_flag) == "e"
    if first_flag == "r":
        return "Routine", _normalize(emergency_flag) == "e"
    if _normalize(emergency_flag) == "e":
        return "Non-Routine", True
    if _normalize(nonroutine_flag) == "nr":
        return "Non-Routine", False
    if _normalize(routine_flag) == "r":
        return "Routine", False
    return "", False


@lru_cache(maxsize=1)
def load_biw_reference() -> dict[str, Any]:
    if not BIW_WORKBOOK_PATH.exists():
        return {"available": False, "activities": [], "hira_rows": []}

    with ZipFile(BIW_WORKBOOK_PATH) as book:
        paths = _sheet_paths(book)
        a8_rows = _rows(book, paths["A8-List of Activities"])
        hira_rows_raw = _rows(book, paths["HIRA"])

    activities: list[BiwActivity] = []
    for row in a8_rows:
        if len(row) < 11 or not row[1].strip().isdigit():
            continue
        routine, emergency = _routine_from_flags(row[8], row[9], row[10])
        activities.append(
            BiwActivity(
                name=row[4].strip(),
                area=row[2].strip(),
                location=row[3].strip(),
                routine=routine,
                emergency=emergency,
            )
        )

    hira_rows: list[HiraReferenceRow] = []
    current_activity = ""
    for row in hira_rows_raw:
        padded = row + [""] * 16
        if padded[0].strip() and padded[0].strip().lower() not in {"normal activity"}:
            current_activity = padded[0].strip()
        if not current_activity or not padded[1].strip() or padded[1].strip().lower() == "type of ohs hazard (a.1)":
            continue
        routine = "Non-Routine" if _normalize(padded[6]) == "nr" else "Routine" if _normalize(padded[6]) == "r" else ""
        direct = "Direct" if _normalize(padded[5]) == "d" else "Indirect" if _normalize(padded[5]) == "i" else ""
        hira_rows.append(
            HiraReferenceRow(
                activity=current_activity,
                hazard_type=padded[1].strip(),
                description=padded[2].strip(),
                affected_people=padded[3].strip(),
                outcome=padded[4].strip(),
                direct_indirect=direct,
                routine_nonroutine=routine,
                overriding_criteria=padded[7].strip(),
                existing_controls=padded[8].strip(),
                gaps=padded[9].strip(),
                likelihood_score=_score(padded[10]),
                scale_score=_score(padded[11]),
                harm_score=_score(padded[12]),
                people_score=_score(padded[13]),
                rpn=_rpn(padded[14]),
                risk_level=padded[15].strip(),
            )
        )

    return {"available": True, "activities": activities, "hira_rows": hira_rows}


def _tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "during",
        "into",
        "from",
        "near",
        "area",
        "task",
        "work",
        "change",
        "changing",
        "changed",
        "replace",
        "replacing",
        "replacement",
        "repair",
        "repairing",
        "maintenance",
    }
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", _normalize(text)):
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if len(token) > 2 and token not in stop:
            tokens.add(token)
    return tokens


def _match_score(query: str, *candidates: str) -> int:
    query_norm = _normalize(query)
    haystack = _normalize(" ".join(candidates))
    if not query_norm or not haystack:
        return 0
    score = 0
    if query_norm in haystack or any(_normalize(candidate) in query_norm for candidate in candidates if candidate):
        score += 20
    score += len(_tokens(query_norm).intersection(_tokens(haystack))) * 4
    return score


def find_activity(query: str) -> BiwActivity | None:
    data = load_biw_reference()
    matches = [
        (_match_score(query, item.name, item.location, item.area), item)
        for item in data["activities"]
    ]
    matches = [item for item in matches if item[0] >= 8]
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def find_hira_rows(query: str, limit: int = 8) -> list[HiraReferenceRow]:
    data = load_biw_reference()
    activity = find_activity(query)
    if activity:
        exact_rows = [
            row
            for row in data["hira_rows"]
            if _normalize(row.activity) == _normalize(activity.name)
        ]
        if exact_rows:
            query_tokens = _tokens(query)
            exact_rows.sort(
                key=lambda row: (
                    len(query_tokens.intersection(_tokens(row.description))),
                    row.likelihood_score or 0,
                    row.harm_score or 0,
                ),
                reverse=True,
            )
            return exact_rows[:limit]

    matches = [
        (_match_score(query, row.activity, row.description, row.hazard_type), row)
        for row in data["hira_rows"]
    ]
    matches = [item for item in matches if item[0] >= 8]
    matches.sort(key=lambda item: item[0], reverse=True)
    rows: list[HiraReferenceRow] = []
    seen: set[tuple[str, str, str]] = set()
    for _, row in matches:
        key = (_normalize(row.activity), _normalize(row.hazard_type), _normalize(row.description))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def compact_biw_context(query: str) -> str:
    activity = find_activity(query)
    rows = find_hira_rows(query, limit=5)
    lines = []
    if activity:
        lines.append(
            f"A8 match: {activity.name}; location={activity.location or 'not specified'}; "
            f"classification={activity.routine or 'not specified'}."
        )
    for row in rows:
        lines.append(
            "HIRA match: "
            f"activity={row.activity}; hazard={row.hazard_type}; desc={row.description}; "
            f"condition={row.direct_indirect}/{row.routine_nonroutine}; "
            f"scores=L{row.likelihood_score}, Scale{row.scale_score}, Harm{row.harm_score}, People{row.people_score}; "
            f"controls={row.existing_controls}; gaps={row.gaps}; risk={row.risk_level}."
        )
    return "\n".join(lines[:6])
