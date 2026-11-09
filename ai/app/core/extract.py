"""본문 추출·정리·문장 분리 (명세서 4.1, 5.1 ①~③)."""

import json
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings
from app.core.errors import InvalidFileError

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MAX_FILE_BYTES = 20 * 1024 * 1024  # Backend에서도 검사하지만 한 번 더 막는다
MAX_DOCX_XML_BYTES = 50 * 1024 * 1024  # 압축을 푼 document.xml 크기 상한 (압축 폭탄 방지)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 머리말·꼬리말로 볼 수 있는 위치: 페이지 맨 위 2줄, 맨 아래 2줄
EDGE_LINES = 2

# 쪽번호만 있는 줄: "3", "- 3 -", "3 / 12", "Page 3"
PAGE_NUMBER_LINE = re.compile(
    r"^(?:-\s*)?\d+(?:\s*-)?$|^\d+\s*/\s*\d+$|^page\s*\d+$", re.IGNORECASE
)
# 문장 끝: "다." "요." "?" "!" 뒤의 공백에서 자른다
SENTENCE_END = re.compile(r"(?:(?<=다\.)|(?<=요\.)|(?<=[?!]))\s+")
MULTI_SPACE = re.compile(r"[ \t\u00a0\u3000]+")


@dataclass
class PageText:
    page_number: int | None  # DOCX는 페이지 개념이 없어 None
    text: str


@dataclass
class ExtractResult:
    content_text: str
    page_count: int | None
    pages: list[PageText]


# ---------- 1. 파일 시그니처 확인 ----------

def check_signature(path: Path, mime_type: str) -> None:
    with path.open("rb") as f:
        head = f.read(5)

    if mime_type == PDF_MIME:
        if head != b"%PDF-":
            raise InvalidFileError("pdf signature mismatch")
        return

    if mime_type == DOCX_MIME:
        # DOCX는 zip 파일이고, 안에 word/document.xml 이 있어야 한다
        if not head.startswith(b"PK\x03\x04"):
            raise InvalidFileError("docx signature mismatch")
        try:
            with zipfile.ZipFile(path) as z:
                if "word/document.xml" not in z.namelist():
                    raise InvalidFileError("docx structure mismatch")
        except zipfile.BadZipFile as e:
            raise InvalidFileError("broken docx") from e
        return

    raise InvalidFileError("unsupported mime type")


# ---------- 2. 로드 ----------

def _load_pdf(path: Path) -> list[PageText]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise InvalidFileError("encrypted pdf")
        return [PageText(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    except InvalidFileError:
        raise
    except Exception as e:
        raise InvalidFileError("pdf read failed") from e


def _run_text(node: ET.Element) -> str:
    """단락·셀 안의 글자를 순서대로 모은다. 탭은 공백, 줄바꿈은 공백으로."""
    parts = []
    for el in node.iter():
        if el.tag == f"{W}t" and el.text:
            parts.append(el.text)
        elif el.tag in (f"{W}tab", f"{W}br", f"{W}cr"):
            parts.append(" ")
    return "".join(parts).strip()


def _table_lines(tbl: ET.Element) -> list[str]:
    """표를 행 단위 한 줄로 바꾼다. 첫 행은 머리글로 보고 각 값 앞에 붙인다.

    예) 머리글 [구분, 사원, 팀장], 행 [숙박비(1박), 70,000원, 90,000원]
        → "숙박비(1박): 사원 70,000원, 팀장 90,000원"
    칸마다 따로 줄을 만들면 '90,000원'이 어느 직급 값인지 사라져서
    검색·답변·버전 비교가 모두 부정확해진다.
    """
    rows = []
    for tr in tbl.findall(f"{W}tr"):
        cells = [_run_text(tc) for tc in tr.findall(f"{W}tc")]
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body or len(header) < 2:
        return [" | ".join(c for c in row if c) for row in rows]

    lines = []
    for row in body:
        label, values = row[0], row[1:]
        pairs = [f"{h} {v}".strip() for h, v in zip(header[1:], values) if v]
        lines.append(f"{label}: {', '.join(pairs)}" if label else ", ".join(pairs))
    return lines


def _load_docx(path: Path) -> list[PageText]:
    """word/document.xml 을 직접 읽어 단락은 한 줄씩, 표는 행 단위로 뽑는다."""
    try:
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise InvalidFileError("docx too large")
            root = ET.fromstring(z.read(info))
    except InvalidFileError:
        raise
    except Exception as e:
        raise InvalidFileError("docx read failed") from e

    body = root.find(f"{W}body")
    lines: list[str] = []
    for child in (body if body is not None else []):
        if child.tag == f"{W}p":
            text = _run_text(child)
            if text:
                lines.append(text)
        elif child.tag == f"{W}tbl":
            lines.extend(_table_lines(child))
    return [PageText(None, "\n".join(lines))]


# ---------- 3. 정리 ----------

def _clean_lines(text: str) -> list[str]:
    """공백을 정리하고 빈 줄만 지운다. (숫자만 있는 줄도 본문일 수 있어 여기서는 지우지 않음)"""
    lines = []
    for raw in text.splitlines():
        line = MULTI_SPACE.sub(" ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _edge_indexes(n: int) -> set[int]:
    """줄 n개 중 맨 위 EDGE_LINES줄, 맨 아래 EDGE_LINES줄의 위치."""
    return set(range(min(EDGE_LINES, n))) | set(range(max(n - EDGE_LINES, 0), n))


def _strip_page_numbers(lines: list[str]) -> list[str]:
    """페이지 맨 위·아래에 있는 쪽번호 줄만 지운다. 본문 중간의 '14' 같은 표 숫자는 남긴다."""
    edges = _edge_indexes(len(lines))
    return [line for i, line in enumerate(lines) if not (i in edges and PAGE_NUMBER_LINE.match(line))]


def _remove_repeated_edge_lines(pages_lines: list[list[str]]) -> list[list[str]]:
    """3쪽 이상 문서에서, 절반 이상의 쪽 '맨 위·아래'에 똑같이 나오는 줄은 머리말·꼬리말로 보고 지운다.

    본문 중간에 반복되는 표 머리글 등은 건드리지 않는다.
    """
    if len(pages_lines) < 3:
        return pages_lines

    counts: Counter[str] = Counter()
    for lines in pages_lines:
        counts.update({lines[i] for i in _edge_indexes(len(lines))})

    threshold = len(pages_lines) / 2
    repeated = {line for line, c in counts.items() if c >= threshold}

    result = []
    for lines in pages_lines:
        edges = _edge_indexes(len(lines))
        result.append([line for i, line in enumerate(lines) if not (i in edges and line in repeated)])
    return result


def split_sentences(lines: list[str]) -> list[str]:
    """한 줄을 문장 단위로 자른다. 문장 부호가 없는 줄(표·목록)은 그대로 한 문장."""
    sentences = []
    for line in lines:
        for part in SENTENCE_END.split(line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


# ---------- 전체 흐름 ----------

def extract(path: Path, mime_type: str) -> ExtractResult:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise InvalidFileError("file too large")

    check_signature(path, mime_type)

    if mime_type == PDF_MIME:
        raw_pages = _load_pdf(path)
        pages_lines = [_clean_lines(p.text) for p in raw_pages]
        pages_lines = _remove_repeated_edge_lines(pages_lines)
        pages_lines = [_strip_page_numbers(lines) for lines in pages_lines]
    else:
        # DOCX는 페이지 구분·쪽번호가 본문에 섞이지 않아서 정리만 한다
        raw_pages = _load_docx(path)
        pages_lines = [_clean_lines(p.text) for p in raw_pages]

    pages = [PageText(p.page_number, "\n".join(lines)) for p, lines in zip(raw_pages, pages_lines)]

    sentences = split_sentences([line for lines in pages_lines for line in lines])
    if not sentences:
        # 스캔 이미지 PDF처럼 텍스트가 없는 경우
        raise InvalidFileError("empty text")

    page_count = len(raw_pages) if mime_type == PDF_MIME else None
    return ExtractResult("\n".join(sentences), page_count, pages)


def save_page_texts(version_id: int, pages: list[PageText]) -> Path:
    """색인 단계에서 쓸 페이지별 텍스트를 잠시 저장한다 (명세서 4.1)."""
    out_dir = Path(settings.VECTOR_STORE_DIR) / "_extract"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{version_id}.json"

    data = [{"pageNumber": p.page_number, "text": p.text} for p in pages]
    # 임시 파일에 다 쓴 뒤 이름을 바꿔서, 쓰다 만 파일이 남지 않게 한다
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, target)
    return target