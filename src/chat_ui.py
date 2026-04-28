"""Streamlit frontend for Tata Motors HIRA chatbot."""

from __future__ import annotations

from datetime import datetime
import json
import os
import re

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from bot_engine import assess_question, create_components

if load_dotenv:
    load_dotenv()


def _split_table_row(row: str) -> list[str]:
    protected = row.strip().strip("|").replace("\\|", "\ue000")
    return [cell.strip().replace("\ue000", "|") for cell in protected.split("|")]


def _is_separator_row(row: str) -> bool:
    cells = _split_table_row(row)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _parse_markdown_report(markdown_text: str) -> list[dict]:
    blocks: list[dict] = []
    lines = markdown_text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue

        if line.startswith("### "):
            blocks.append({"type": "heading", "text": line[4:].strip()})
            index += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1

            if not table_lines:
                continue

            blocks.append(
                {
                    "type": "table",
                    "header": _split_table_row(table_lines[0]),
                    "rows": [
                        _split_table_row(row)
                        for row in table_lines[1:]
                        if not _is_separator_row(row)
                    ],
                }
            )
            continue

        blocks.append({"type": "paragraph", "text": line.replace("**", "")})
        index += 1

    return blocks


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_text(text: str, max_width: float, font_size: int) -> list[str]:
    max_chars = max(8, int(max_width / (font_size * 0.48)))
    words = re.sub(r"\s+", " ", str(text).strip()).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                lines.append(current)
                current = ""
            for start in range(0, len(word), max_chars):
                lines.append(word[start : start + max_chars])
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _pdf_text(x: float, y: float, text: str, size: int = 9, bold: bool = False, color: tuple[float, float, float] = (0.06, 0.13, 0.2)) -> str:
    font = "F2" if bold else "F1"
    return f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET\n"


def _pdf_rect(x: float, y: float, width: float, height: float, fill: tuple[float, float, float] | None = None, stroke: tuple[float, float, float] = (0.80, 0.86, 0.91)) -> str:
    commands = ""
    if fill:
        commands += f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f\n"
    commands += f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG {x:.2f} {y:.2f} {width:.2f} {height:.2f} re S\n"
    return commands


def _markdown_report_to_pdf(markdown_text: str) -> bytes:
    """Render the generated Markdown tables as a simple tabular PDF report."""
    blocks = _parse_markdown_report(markdown_text)
    page_width = 842.0
    page_height = 595.0
    margin = 34.0
    content_width = page_width - (margin * 2)
    bottom = 34.0
    row_padding = 7.0
    line_height = 11.0
    pages: list[str] = []
    stream = ""
    y = page_height - margin

    def new_page() -> None:
        nonlocal stream, y
        if stream:
            pages.append(stream)
        stream = ""
        y = page_height - margin

    def ensure_space(height: float) -> None:
        nonlocal y
        if y - height < bottom:
            new_page()

    generated_at = datetime.now().strftime("%d-%m-%Y %H:%M")
    stream += _pdf_text(margin, y - 4, "TML HIRA Report", size=18, bold=True, color=(0.07, 0.24, 0.41))
    y -= 22
    stream += _pdf_text(margin, y, f"Generated: {generated_at}", size=8, color=(0.37, 0.42, 0.48))
    y -= 20

    for block in blocks:
        if block["type"] == "heading":
            ensure_space(28)
            stream += _pdf_text(margin, y, block["text"], size=13, bold=True, color=(0.07, 0.24, 0.41))
            y -= 16
            continue

        if block["type"] == "paragraph":
            lines = _wrap_pdf_text(block["text"], content_width, 9)
            ensure_space(len(lines) * line_height + 8)
            for line in lines:
                stream += _pdf_text(margin, y, line, size=9)
                y -= line_height
            y -= 6
            continue

        if block["type"] != "table":
            continue

        header = block["header"]
        rows = block["rows"]
        column_count = max(1, len(header))
        if column_count == 2:
            widths = [content_width * 0.28, content_width * 0.72]
        elif column_count == 3:
            widths = [content_width * 0.20, content_width * 0.24, content_width * 0.56]
        else:
            widths = [content_width / column_count] * column_count

        def draw_row(cells: list[str], is_header: bool = False, shaded: bool = False) -> None:
            nonlocal stream, y
            wrapped = [
                _wrap_pdf_text(cells[index] if index < len(cells) else "", widths[index] - 12, 8 if not is_header else 8)
                for index in range(column_count)
            ]
            row_height = max(len(cell_lines) for cell_lines in wrapped) * line_height + row_padding * 2
            ensure_space(row_height + (14 if is_header else 0))
            x = margin
            cell_y = y - row_height
            fill = (0.91, 0.95, 0.98) if is_header else ((0.97, 0.98, 0.99) if shaded else None)
            for index, width in enumerate(widths):
                stream += _pdf_rect(x, cell_y, width, row_height, fill=fill)
                text_y = y - row_padding - 8
                for line in wrapped[index]:
                    stream += _pdf_text(x + 6, text_y, line, size=8, bold=is_header or index == 0)
                    text_y -= line_height
                x += width
            y -= row_height

        draw_row(header, is_header=True)
        for row_index, row in enumerate(rows):
            draw_row(row, shaded=row_index % 2 == 1)
        y -= 14

    if stream:
        pages.append(stream)

    bold_font_obj = 4 + len(pages) * 2
    objects = [
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + index * 2} 0 R' for index in range(len(pages)))}] /Count {len(pages)} >>",
        "<< /Type /Catalog /Pages 2 0 R >>",
    ]
    kids = []
    for page_index, content in enumerate(pages):
        page_obj = 4 + page_index * 2
        content_obj = page_obj + 1
        kids.append(f"{page_obj} 0 R")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.0f} {page_height:.0f}] "
            f"/Resources << /Font << /F1 1 0 R /F2 {bold_font_obj} 0 R >> >> /Contents {content_obj} 0 R >>"
        )
        content_bytes = content.encode("latin-1", errors="replace")
        objects.append(f"<< /Length {len(content_bytes)} >>\nstream\n{content}\nendstream")

    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_index} 0 obj\n{obj}\nendobj\n".encode("latin-1", errors="replace"))

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 3 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("ascii")
    )
    return bytes(pdf)


def _report_filename(state: dict | None) -> str:
    activity = ""
    if isinstance(state, dict):
        activity = str(state.get("activity") or "")
    slug = re.sub(r"[^a-z0-9]+", "-", activity.lower()).strip("-")[:45] or "hira-report"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{slug}-{timestamp}.pdf"


st.set_page_config(page_title="Tata Motors HIRA Bot", page_icon="TM", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --tml-blue: #123c69;
        --tml-teal: #0f766e;
        --ink: #102033;
        --muted: #5f6b7a;
        --line: #d8e1ea;
        --soft: #f4f8fb;
        --warning: #f59e0b;
    }

    .stApp {
        background:
            linear-gradient(180deg, #f7fbff 0%, #eef5f9 52%, #eaf0f5 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1120px;
        padding: 44px 32px 120px;
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(18, 60, 105, 0.98), rgba(15, 118, 110, 0.95)),
            linear-gradient(90deg, #123c69, #0f766e);
        border: 1px solid rgba(255, 255, 255, 0.24);
        border-radius: 8px;
        box-shadow: 0 18px 50px rgba(31, 54, 82, 0.20);
        color: #ffffff;
        padding: 30px 34px;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 38px;
        line-height: 1.15;
        margin: 0;
        letter-spacing: 0;
    }

    .hero p {
        color: rgba(255, 255, 255, 0.88);
        font-size: 17px;
        margin: 12px 0 0;
        max-width: 780px;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 8px 22px rgba(31, 54, 82, 0.07);
        margin-bottom: 12px;
        padding: 8px 10px;
    }

    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"] * {
        color: var(--ink) !important;
        letter-spacing: 0 !important;
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        font-size: 16px;
        line-height: 1.55;
    }

    [data-testid="stChatMessageContent"] h3 {
        color: var(--tml-blue) !important;
        font-size: 18px;
        margin: 16px 0 8px;
    }

    [data-testid="stChatMessageContent"] table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        margin: 10px 0 22px;
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        font-size: 15px;
    }

    [data-testid="stChatMessageContent"] th,
    [data-testid="stChatMessageContent"] td {
        border: 1px solid var(--line);
        padding: 10px 12px;
        text-align: left !important;
        vertical-align: top;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
        line-height: 1.45;
    }

    [data-testid="stChatMessageContent"] th {
        background: #e7f0f6;
        color: var(--tml-blue) !important;
        font-weight: 700;
    }

    [data-testid="stChatMessageContent"] td {
        background: #ffffff;
    }

    [data-testid="stChatMessageContent"] tr:nth-child(even) td {
        background: #f8fbfd;
    }

    [data-testid="stChatMessageContent"] table th:first-child,
    [data-testid="stChatMessageContent"] table td:first-child {
        width: 24%;
        font-weight: 650;
    }

    [data-testid="stChatMessageContent"] code {
        background: #edf4f8 !important;
        border: 1px solid #cddbe7;
        color: #0b2545 !important;
        border-radius: 5px;
        padding: 2px 6px;
        font-size: 14px;
    }

    [data-testid="stBottomBlockContainer"] {
        background: rgba(14, 22, 34, 0.92);
        border-top: 1px solid rgba(255, 255, 255, 0.10);
        padding: 16px 0;
    }

    [data-testid="stChatInput"] {
        max-width: 1120px;
        margin: 0 auto;
    }

    [data-testid="stChatInput"] div,
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #0f172a !important;
        caret-color: #0f172a !important;
    }

    [data-testid="stChatInput"] textarea {
        border: 1px solid #bfd0df !important;
        border-radius: 8px !important;
        min-height: 58px !important;
        font-size: 16px !important;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #536170 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInputSubmitButton"] {
        background: var(--tml-teal) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    [data-testid="stDownloadButton"] {
        margin: 12px 0 2px;
    }

    [data-testid="stDownloadButton"] button {
        background: var(--tml-teal) !important;
        color: #ffffff !important;
        border: 1px solid #0d9488 !important;
        border-radius: 8px !important;
        min-height: 42px;
        padding: 0 18px;
        font-size: 15px;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.22);
    }

    [data-testid="stDownloadButton"] button:hover {
        background: #0b665f !important;
        border-color: #0b665f !important;
        color: #ffffff !important;
    }

    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span {
        color: #ffffff !important;
    }

    .stSpinner, .stSpinner * {
        color: var(--ink) !important;
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 22px 14px 120px;
        }
        .hero {
            padding: 22px 20px;
        }
        .hero h1 {
            font-size: 28px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_components(model_name: str):
    os.environ["GEMINI_MODEL"] = model_name
    return create_components()


CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.markdown(
    """
    <section class="hero">
        <h1>Tata Motors HIRA Safety Chatbot</h1>
        <p>Describe the process or activity and the assistant will prepare a TML HIRA draft using the supplied standard, deterministic RPN scoring, and Gemini API extraction.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Can you describe the work activity you are planning to perform?"
            ),
        }
    ]

if "hira_state" not in st.session_state:
    st.session_state.hira_state = {}

if not GEMINI_API_KEY:
    st.warning("Set GEMINI_API_KEY in your environment to enable Gemini extraction for each query.")

for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("report_pdf"):
            st.download_button(
                "Download HIRA report PDF",
                data=msg["report_pdf"],
                file_name=msg.get("report_filename", "hira-report.pdf"),
                mime="application/pdf",
                key=f"download-report-{msg_index}",
            )

user_task = st.chat_input("Describe your task or answer the current HIRA question")

if user_task:
    st.session_state.messages.append({"role": "user", "content": user_task})
    with st.chat_message("user"):
        st.markdown(user_task)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing task and preparing next HIRA step..."):
            llm = get_components(CHAT_MODEL)
            result = assess_question(
                llm=llm,
                user_question=user_task,
                additional_details=json.dumps(st.session_state.hira_state),
            )

        if isinstance(result.get("state"), dict):
            st.session_state.hira_state = result["state"]

        st.markdown(result["answer"])
        assistant_message = {"role": "assistant", "content": result["answer"]}
        if result.get("completed"):
            report_pdf = _markdown_report_to_pdf(result["answer"])
            report_filename = _report_filename(result.get("state"))
            assistant_message["report_pdf"] = report_pdf
            assistant_message["report_filename"] = report_filename
            st.download_button(
                "Download HIRA report PDF",
                data=report_pdf,
                file_name=report_filename,
                mime="application/pdf",
                key=f"download-report-new-{len(st.session_state.messages)}",
            )

        st.session_state.messages.append(assistant_message)
