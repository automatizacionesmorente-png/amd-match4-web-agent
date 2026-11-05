import csv
import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET

TEXT_EXT = {".txt", ".log", ".md", ".py", ".json", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".toml", ".xml",
            ".html", ".htm", ".sh", ".js", ".ts", ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rs", ".sql", ".rst",
            ".properties", ".env.example", ".tex"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


class Unreadable(Exception):
    pass


def _read_bytes(path, limit=50_000_000):
    with open(path, "rb") as f:
        return f.read(limit)


def _looks_text(data):
    if not data:
        return False
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        s = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in s)
    return printable / max(1, len(s)) > 0.95


def _rows_to_text(rows):
    rows = [[("" if c is None else str(c)).strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    header = rows[0]
    out = [" | ".join(header)]
    for r in rows[1:]:
        if len(header) == len(r) and any(header):
            out.append(" | ".join(f"{h}: {v}" if h else v for h, v in zip(header, r)))
        else:
            out.append(" | ".join(r))
    return "\n".join(out)


def parse_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    # Any encrypted PDF counts as unreadable: a file we were not given the password for is never a source.
    if reader.is_encrypted:
        raise Unreadable("encrypted pdf")
    pages, images = [], []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        pages.append(f"[page {i + 1}]\n{t}")
        if len(t.strip()) < 40:
            try:
                for im in page.images:
                    images.append(im.data)
            except Exception:
                pass
    return "\n".join(pages), images


def parse_docx(path):
    images = []
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        for name in z.namelist():
            if name.startswith("word/media/"):
                images.append(z.read(name))
    body = root.find(f"{W}body")
    out = []
    for el in (body if body is not None else []):
        if el.tag == f"{W}p":
            t = "".join(x.text or "" for x in el.iter(f"{W}t"))
            if t.strip():
                out.append(t)
        elif el.tag == f"{W}tbl":
            rows = []
            for tr in el.iter(f"{W}tr"):
                rows.append(["".join(x.text or "" for x in tc.iter(f"{W}t")) for tc in tr.findall(f"{W}tc")])
            out.append(_rows_to_text(rows))
    return "\n".join(out), images


def parse_pptx(path):
    out = []
    with zipfile.ZipFile(path) as z:
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        for i, n in enumerate(slides):
            root = ET.fromstring(z.read(n))
            out.append(f"[slide {i + 1}]\n" + "\n".join(x.text or "" for x in root.iter(f"{A}t")))
    return "\n".join(out), []


def parse_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        out.append(f"[sheet {ws.title}]\n" + _rows_to_text(rows))
    wb.close()
    return "\n".join(out), []


def parse_csv(path):
    raw = _read_bytes(path).decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return _rows_to_text(list(csv.reader(io.StringIO(raw), dialect))), []


def parse_file(path):
    """Return (kind, text, embedded_images). Raises on anything unreadable."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        _read_bytes(path, 16)
        return "image", "", []
    if ext == ".pdf":
        t, ims = parse_pdf(path)
        return "pdf", t, ims
    if ext == ".docx":
        t, ims = parse_docx(path)
        return "docx", t, ims
    if ext == ".pptx":
        t, ims = parse_pptx(path)
        return "pptx", t, ims
    if ext in (".xlsx", ".xlsm"):
        t, ims = parse_xlsx(path)
        return "xlsx", t, ims
    if ext in (".csv", ".tsv"):
        t, ims = parse_csv(path)
        return "csv", t, ims
    data = _read_bytes(path)
    if ext in TEXT_EXT or _looks_text(data):
        return "text", data.decode("utf-8", errors="replace"), []
    raise Unreadable(f"unknown type {ext}")


def walk(corpus):
    """Yield (relpath, abspath) for every file, never letting one bad entry stop the walk."""
    stack = [corpus]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file():
                    yield os.path.relpath(e.path, corpus).replace(os.sep, "/"), e.path
            except OSError:
                continue
