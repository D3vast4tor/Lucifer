"""Local, bounded document extraction. Document content is never executed."""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import io
import math
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from urllib.parse import unquote, urlsplit

from defusedxml import ElementTree


MAX_BYTES = 128 * 1024 * 1024
MAX_TEXT = 8 * 1024 * 1024
MAX_PAGES = 1000
MAX_VISUALS = 24
MAX_ROWS = 100000
MAX_CELLS = 2000000
CHUNK_CHARS = 2400
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".ico"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpeg", ".mpg", ".ogv"}


@dataclass
class ExtractedDocument:
    sections: list[dict] = field(default_factory=list)
    visuals: list[tuple[str, bytes]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    _text_bytes: int = field(default=0, repr=False)
    _visual_hashes: set[str] = field(default_factory=set, repr=False)

    def add(self, label, text):
        if text and text.strip():
            size = len(text.encode("utf-8"))
            if self._text_bytes + size > MAX_TEXT:
                raise ValueError("Extracted text exceeds 8 MiB; split the document before attaching it")
            self.sections.append({"label": label, "text": text, "authority": "literal"})
            self._text_bytes += size

    def visual(self, label, raw):
        signature = hashlib.sha256(raw).hexdigest()
        if signature in self._visual_hashes:
            return
        if len(self.visuals) >= MAX_VISUALS:
            raise ValueError("Document exceeds 24 visual inputs; split it for complete visual extraction")
        self.visuals.append((label, raw))
        self._visual_hashes.add(signature)


def image_bytes(image):
    from PIL import ImageOps
    if image.width * image.height > 40_000_000:
        raise ValueError("Image exceeds forty million pixels; resize it before attaching")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1536, 1536))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88)
    return output.getvalue()


def _zip(raw):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    entries = archive.infolist()
    if len(entries) > 10000 or sum(e.file_size for e in entries) > 128 * 1024 * 1024:
        archive.close()
        raise ValueError("Document archive exceeds extraction bounds")
    if any(entry.flag_bits & 1 for entry in entries):
        archive.close()
        raise ValueError("Document archive is encrypted; attach an unlocked copy")
    return archive


def _xml_text(raw):
    root = ElementTree.fromstring(raw)
    return "\n".join(" ".join(node.itertext()).strip() for node in root.iter()
                     if node.tag.rsplit("}", 1)[-1] in {"p", "h", "tr"})


def _decode(raw):
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encodings = ["utf-32"]
    elif raw[:2] in {b"\xff\xfe", b"\xfe\xff"}:
        encodings = ["utf-16"]
    else:
        encodings = ["utf-8-sig", "cp1252"]
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            if "\x00" not in text and not any(ord(char) < 32 and char not in "\n\r\t\f" for char in text):
                return text
        except UnicodeError:
            continue
    raise ValueError("Unsupported binary or text encoding; export this document as UTF-8, PDF or Office format")


def _natural(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _local(node):
    return node.tag.rsplit("}", 1)[-1]


def _attribute(node, name, default=""):
    return next((value for key, value in node.attrib.items() if key.rsplit("}", 1)[-1] == name), default)


def _word_text(node):
    text = []
    for child in node.iter():
        kind = _local(child)
        if kind in {"t", "instrText"}:
            text.append(child.text or "")
        elif kind == "tab":
            text.append("\t")
        elif kind in {"br", "cr"}:
            text.append("\n")
        elif kind == "p" and text:
            text.append("\n")
    return "".join(text)


def _word_blocks(node):
    for child in node:
        kind = _local(child)
        if kind == "p":
            yield _word_text(child)
        elif kind == "tbl":
            for index, row in enumerate((x for x in child if _local(x) == "tr"), 1):
                yield f"row {index}: " + "\t".join(_word_text(cell) for cell in row if _local(cell) == "tc")
        else:
            yield from _word_blocks(child)


def _archive_visuals(archive, result, prefixes):
    from PIL import Image
    for member in sorted(archive.namelist(), key=_natural):
        if member.endswith("/") or not any(member.startswith(prefix) for prefix in prefixes):
            continue
        suffix = Path(member).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            with Image.open(io.BytesIO(archive.read(member))) as image:
                result.visual(member, image_bytes(image))
        else:
            result.warnings.append(f"Embedded visual {member} cannot be decoded as a raster image; export it to PDF or PNG for visual inspection.")


def _odf_inline(node):
    parts = [node.text or ""]
    for child in node:
        kind = _local(child)
        if kind == "s":
            count = int(_attribute(child, "c", "1"))
            if not 1 <= count <= MAX_TEXT:
                raise ValueError("ODF whitespace expansion exceeds extraction bounds")
            parts.append(" " * count)
        elif kind == "tab":
            parts.append("\t")
        elif kind == "line-break":
            parts.append("\n")
        else:
            parts.append(_odf_inline(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _odf_extract(archive, suffix, result):
    root = ElementTree.fromstring(archive.read("content.xml"))
    if suffix == ".ods":
        for sheet in (node for node in root.iter() if _local(node) == "table"):
            row_number, lines, text_size = 1, [], 0
            for row in (node for node in sheet.iter() if _local(node) == "table-row"):
                row_count = int(_attribute(row, "number-rows-repeated", "1"))
                if not 1 <= row_count <= 1048576:
                    raise ValueError("ODF repeated row count exceeds spreadsheet bounds")
                cells, column = [], 1
                for cell in row:
                    if _local(cell) not in {"table-cell", "covered-table-cell"}:
                        continue
                    repeated = int(_attribute(cell, "number-columns-repeated", "1"))
                    if not 1 <= repeated <= 16384 or column + repeated - 1 > 16384:
                        raise ValueError("ODF column count exceeds spreadsheet bounds")
                    paragraphs = [_odf_inline(p) for p in cell.iter() if _local(p) in {"p", "h"}]
                    value = "\n".join(paragraphs) or _attribute(cell, "value") or _attribute(cell, "date-value") or _attribute(cell, "boolean-value")
                    formula = _attribute(cell, "formula")
                    if formula:
                        value = formula + (f" [cached value: {value}]" if value else "")
                    if value:
                        label = str(column) if repeated == 1 else f"{column}..{column + repeated - 1} (repeated)"
                        cells.append(f"column {label}: {value}")
                    column += repeated
                if cells:
                    label = str(row_number) if row_count == 1 else f"{row_number}..{row_number + row_count - 1} (repeated)"
                    line = f"row {label}: " + "\t".join(cells)
                    text_size += len(line.encode("utf-8"))
                    if result._text_bytes + text_size > MAX_TEXT:
                        raise ValueError("Spreadsheet exceeds text extraction bounds")
                    lines.append(line)
                row_number += row_count
            result.add(f"sheet {_attribute(sheet, 'name', 'unnamed')}", "\n".join(lines))
        result.warnings.append("Formula text and stored cell values are preserved; spreadsheet formulas were not executed.")
    elif suffix == ".odp":
        for number, page in enumerate((node for node in root.iter() if _local(node) == "page"), 1):
            result.add(f"slide {number}", "\n".join(_odf_inline(node) for node in page.iter() if _local(node) in {"p", "h"}))
    else:
        result.add("document", "\n".join(_odf_inline(node) for node in root.iter() if _local(node) in {"p", "h"}))
    _archive_visuals(archive, result, ("Pictures/",))


def _epub_order(archive):
    names = archive.namelist()
    html = [name for name in names if name.lower().endswith((".xhtml", ".html", ".htm"))]
    if "META-INF/container.xml" not in names:
        raise ValueError("EPUB is missing its reading-order container")
    root = ElementTree.fromstring(archive.read("META-INF/container.xml"))
    package = next((_attribute(node, "full-path") for node in root.iter() if _local(node) == "rootfile"), None)
    if not package or package not in names:
        raise ValueError("EPUB is missing its package manifest")
    package_root = ElementTree.fromstring(archive.read(package))
    manifest = {_attribute(node, "id"): _attribute(node, "href")
                for node in package_root.iter() if _local(node) == "item"}
    ordered = []
    for item in package_root.iter():
        if _local(item) == "itemref":
            href = manifest.get(_attribute(item, "idref"), "")
            member = posixpath.normpath(posixpath.join(posixpath.dirname(package), unquote(urlsplit(href).path)))
            if member in html and member not in ordered:
                ordered.append(member)
    return ordered + [name for name in html if name not in ordered]


def _relationships(archive, member):
    relpath = posixpath.join(posixpath.dirname(member), "_rels", posixpath.basename(member) + ".rels")
    if relpath not in archive.namelist():
        return {}
    root = ElementTree.fromstring(archive.read(relpath))
    result = {}
    for node in root:
        target = _attribute(node, "Target")
        if _attribute(node, "TargetMode") == "External" or urlsplit(target).scheme:
            continue
        path = posixpath.normpath(posixpath.join(posixpath.dirname(member), unquote(target)))
        if path in archive.namelist():
            result[_attribute(node, "Id")] = path
    return result


def _presentation_order(archive):
    names = archive.namelist()
    ordered = []
    if "ppt/presentation.xml" in names:
        links = _relationships(archive, "ppt/presentation.xml")
        root = ElementTree.fromstring(archive.read("ppt/presentation.xml"))
        for node in root.iter():
            if _local(node) == "sldId":
                relation = next((value for key, value in node.attrib.items()
                                 if key.endswith("relationships}id")), "")
                if links.get(relation) and links[relation] not in ordered:
                    ordered.append(links[relation])
    ordered += [name for name in sorted(names, key=_natural)
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name) and name not in ordered]
    with_notes = []
    for slide in ordered:
        with_notes.append(slide)
        with_notes += [member for member in _relationships(archive, slide).values()
                       if re.fullmatch(r"ppt/(notesSlides/notesSlide|charts/chart)\d+\.xml", member)
                       and member not in with_notes]
    return with_notes + [name for name in sorted(names, key=_natural)
                         if re.fullmatch(r"ppt/(notesSlides/notesSlide|charts/chart)\d+\.xml", name)
                         and name not in with_notes]


def _video(raw, result, frames, suffix):
    import av
    started = time.monotonic()
    formats = {".mp4": "mov", ".m4v": "mov", ".mov": "mov", ".mkv": "matroska", ".webm": "matroska",
               ".avi": "avi", ".mpeg": "mpeg", ".mpg": "mpeg", ".ogv": "ogg"}
    # Filename-selected demuxers and pipe-only I/O prevent a disguised HLS/DASH
    # playlist from following URLs or opening other local files during probing.
    with av.open(io.BytesIO(raw), format=formats[suffix], options={"protocol_whitelist": "pipe"}) as container:
        videos = container.streams.video
        if not videos:
            raise ValueError("No readable video stream; attach a transcript for audio-only media")
        stream = videos[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else (
            float(container.duration / av.time_base) if container.duration else None)
        if not duration or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Video duration is unavailable; remux the file or attach extracted images")
        times = [duration * (i + 0.5) / frames for i in range(frames)]
        seen = set()
        start_time = float((stream.start_time or 0) * stream.time_base)
        for position in times:
            if time.monotonic() - started > 60:
                raise ValueError("Video extraction exceeded sixty seconds; split or reduce the video")
            container.seek(int((position + start_time) / stream.time_base), stream=stream, backward=True)
            candidate, timestamp, previous = None, position, None
            for index, frame in enumerate(container.decode(stream)):
                if index > 600 or time.monotonic() - started > 60:
                    raise ValueError("Video seek could not reach the requested frame within extraction bounds")
                timestamp = max(0.0, float(frame.time) - start_time) if frame.time is not None else position
                candidate = frame
                if timestamp < position:
                    previous = (frame, timestamp)
                    continue
                if previous is not None and position - previous[1] < timestamp - position:
                    candidate, timestamp = previous
                break
            # A target in the final frame's display interval has no later frame.
            # Keep the last decoded candidate rather than silently dropping it.
            if candidate is not None:
                key = round(timestamp, 3)
                if key not in seen:
                    result.visuals.append((f"frame at {timestamp:.3f}s", image_bytes(candidate.to_image())))
                    seen.add(key)
        if not seen:
            raise ValueError("Video contained no decodable frames")
        result.metadata.update(duration_seconds=duration, sampled_frames=len(seen), audio_present=bool(container.streams.audio))
        result.warnings.append("Visual coverage is sampled across the video, not every frame. Audio is not transcribed; attach subtitles or a transcript for spoken content.")
        # Decode embedded text subtitles where supported; image subtitles remain visual data.
        if container.streams.subtitles:
            container.seek(0)
            subtitle_text, subtitle_bytes, bitmap_subtitles, errors = [], 0, False, 0
            for packet in container.demux(*container.streams.subtitles):
                if time.monotonic() - started > 60:
                    raise ValueError("Embedded subtitles exceeded sixty seconds; attach a subtitle file separately")
                try:
                    for subtitle in packet.decode():
                        for rectangle in getattr(subtitle, "rects", (subtitle,)):
                            text = getattr(rectangle, "dialogue", None) or getattr(rectangle, "text", None)
                            if isinstance(text, bytes):
                                text = text.decode("utf-8", errors="replace")
                            if text:
                                timestamp = float(packet.pts * packet.time_base) if packet.pts is not None and packet.time_base else 0.0
                                line = f"[{timestamp:.3f}s] {text}"
                                subtitle_bytes += len(line.encode("utf-8"))
                                if result._text_bytes + subtitle_bytes > MAX_TEXT:
                                    raise ValueError("Video subtitles exceed text bounds")
                                subtitle_text.append(line)
                            elif getattr(rectangle, "type", None) in {b"bitmap", "bitmap"}:
                                bitmap_subtitles = True
                except (TypeError, AttributeError, av.error.FFmpegError):
                    errors += 1
                    continue
            result.add("embedded subtitles", "\n".join(subtitle_text))
            if bitmap_subtitles:
                result.warnings.append("Embedded bitmap subtitles were not OCR-transcribed; attach text subtitles for complete spoken content.")
            if errors:
                result.warnings.append(f"{errors} embedded subtitle packets could not be decoded; attach a text transcript for complete spoken content.")


def extract_document(name: str, raw: bytes, *, video_frames=8) -> ExtractedDocument:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Attach a nonempty document no larger than 128 MiB")
    if not 1 <= video_frames <= MAX_VISUALS:
        raise ValueError("Video frame count must be 1..24")
    result, suffix = ExtractedDocument(), Path(name).suffix.lower()
    for signature, detected in [(b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"),
                                 (b"GIF87a", ".gif"), (b"GIF89a", ".gif"), (b"II*\x00", ".tif"),
                                 (b"MM\x00*", ".tif"), (b"\x00\x00\x01\x00", ".ico")]:
        if raw.startswith(signature):
            suffix = detected
            break
    if len(raw) >= 14 and raw.startswith(b"BM") and 14 <= int.from_bytes(raw[2:6], "little") <= len(raw):
        suffix = ".bmp"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        suffix = ".webp"
    if raw.startswith(b"PK") and suffix not in {".xlsx", ".xlsm", ".xltx", ".xltm", ".docx", ".docm", ".pptx", ".pptm", ".odt", ".ods", ".odp", ".epub"}:
        with _zip(raw) as archive:
            names = set(archive.namelist())
            for marker, detected in [("word/document.xml", ".docx"), ("xl/workbook.xml", ".xlsx"),
                                     ("ppt/presentation.xml", ".pptx"), ("META-INF/container.xml", ".epub")]:
                if marker in names:
                    suffix = detected
                    break
            else:
                if "mimetype" in names:
                    mime = archive.read("mimetype").decode("ascii", errors="ignore")
                    suffix = {"application/vnd.oasis.opendocument.text": ".odt", "application/vnd.oasis.opendocument.spreadsheet": ".ods", "application/vnd.oasis.opendocument.presentation": ".odp"}.get(mime, suffix)
    try:
        if raw.startswith(b"%PDF-") or suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF is encrypted; attach an unlocked copy")
            if len(reader.pages) > MAX_PAGES:
                raise ValueError("PDF exceeds 1000 pages; split it before attaching")
            result.metadata["pages"] = len(reader.pages)
            for number, page in enumerate(reader.pages, 1):
                content = page.get_contents()
                if content is not None and len(content.get_data()) > 16 * 1024 * 1024:
                    raise ValueError(f"PDF page {number} has an oversized content stream")
                text = page.extract_text() or ""
                result.add(f"page {number}", text)
                if not text.strip() or len(page.images):
                    if len(result.visuals) >= MAX_VISUALS:
                        raise ValueError("PDF has more than 24 scanned pages; split it for complete visual extraction")
                    if not shutil.which("pdftoppm"):
                        raise ValueError("Scanned PDF needs pdftoppm and a loaded vision model")
                    rendered = subprocess.run(["pdftoppm", "-f", str(number), "-l", str(number),
                        "-scale-to", "1536", "-singlefile", "-png", "-"], input=raw,
                        capture_output=True, timeout=30, check=True)
                    from PIL import Image
                    result.visual(f"page {number} visual", image_bytes(Image.open(io.BytesIO(rendered.stdout))))
            result.warnings.append("PDF text is extracted in source reading order; complex vector diagrams and exact page layout may require a page image for visual inspection.")
        elif suffix in IMAGE_EXTENSIONS:
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as image:
                count = getattr(image, "n_frames", 1)
                if suffix in {".tiff", ".tif"}:
                    if count > MAX_VISUALS:
                        raise ValueError("Multipage image exceeds 24 pages; split it before attaching")
                    indices = list(range(count))
                else:
                    indices = sorted({min(count - 1, int(count * (index + 0.5) / min(video_frames, count)))
                                      for index in range(min(video_frames, count))})
                for index in indices:
                    image.seek(index)
                    result.visual(f"image frame {index + 1}" if count > 1 else "image", image_bytes(image))
                result.metadata["image_frames"] = count
                if count > len(indices):
                    result.warnings.append(f"Animated image: sampled {len(indices)} of {count} frames; intermediate motion is not fully represented.")
        elif suffix in VIDEO_EXTENSIONS:
            _video(raw, result, video_frames, suffix)
        elif suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
            with _zip(raw):
                pass
            from openpyxl import load_workbook
            book = load_workbook(io.BytesIO(raw), read_only=True, data_only=False, keep_links=False)
            cached_book = load_workbook(io.BytesIO(raw), read_only=True, data_only=True, keep_links=False)
            try:
                for sheet in book:
                    lines, text_size, cell_count = [], 0, 0
                    cached_rows = cached_book[sheet.title].iter_rows(values_only=True)
                    for index, row in enumerate(sheet.iter_rows(values_only=True), 1):
                        if index > MAX_ROWS:
                            raise ValueError("Spreadsheet exceeds 100000 rows per sheet; split it before attaching")
                        cell_count += len(row)
                        if cell_count > MAX_CELLS:
                            raise ValueError("Spreadsheet exceeds two million expanded cells; remove unused formatted ranges or split it")
                        cached_row = next(cached_rows, ())
                        values = []
                        for column, cell in enumerate(row):
                            value = "" if cell is None else str(cell)
                            cached_value = cached_row[column] if column < len(cached_row) else None
                            if isinstance(cell, str) and cell.startswith("=") and cached_value is not None:
                                value += f" [cached value: {cached_value}]"
                            values.append(value)
                        line = f"row {index}: " + "\t".join(values)
                        text_size += len(line.encode("utf-8")) + 1
                        if result._text_bytes + text_size > MAX_TEXT:
                            raise ValueError("Spreadsheet exceeds text extraction bounds")
                        lines.append(line)
                    result.add(f"sheet {sheet.title}", "\n".join(lines))
                result.warnings.append("Spreadsheet formulas and available cached values are preserved as source text; no formulas or macros were executed.")
            finally:
                book.close()
                cached_book.close()
        elif suffix == ".xls":
            import xlrd
            book = xlrd.open_workbook(file_contents=raw, on_demand=True)
            try:
                for sheet in book.sheets():
                    if sheet.nrows > MAX_ROWS or sheet.nrows * sheet.ncols > MAX_CELLS:
                        raise ValueError("Legacy spreadsheet exceeds row or cell extraction bounds")
                    lines, text_size = [], 0
                    for index in range(sheet.nrows):
                        values = []
                        for cell in sheet.row(index):
                            value = cell.value
                            if cell.ctype == xlrd.XL_CELL_DATE:
                                value = xlrd.xldate_as_datetime(value, book.datemode).isoformat()
                            elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                                value = bool(value)
                            elif cell.ctype == xlrd.XL_CELL_ERROR:
                                value = xlrd.error_text_from_code.get(value, f"cell error {value}")
                            values.append(str(value))
                        line = f"row {index + 1}: " + "\t".join(values)
                        text_size += len(line.encode("utf-8")) + 1
                        if result._text_bytes + text_size > MAX_TEXT:
                            raise ValueError("Spreadsheet exceeds text extraction bounds")
                        lines.append(line)
                    result.add(f"sheet {sheet.name}", "\n".join(lines))
                result.warnings.append("Legacy XLS uses stored cell results; formula definitions are unavailable and no macros were executed.")
            finally:
                book.release_resources()
        elif suffix in {".docx", ".docm"}:
            with _zip(raw) as archive:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
                for index, text in enumerate(_word_blocks(root), 1):
                    result.add(f"block {index}", text)
                for member in sorted(archive.namelist(), key=_natural):
                    if re.fullmatch(r"word/(header\d+|footer\d+|footnotes|endnotes|comments)\.xml", member):
                        part = ElementTree.fromstring(archive.read(member))
                        result.add(member, "\n".join(_word_text(node) for node in part.iter() if _local(node) == "p"))
                _archive_visuals(archive, result, ("word/media/",))
                if any(member.startswith("word/embeddings/") for member in archive.namelist()):
                    result.warnings.append("Embedded Office objects are not executed; attach their source files separately for complete object data.")
        elif suffix in {".pptx", ".pptm", ".odt", ".ods", ".odp", ".epub"}:
            with _zip(raw) as archive:
                if suffix.startswith(".ppt"):
                    for member in _presentation_order(archive):
                        root = ElementTree.fromstring(archive.read(member))
                        result.add(member, "\n".join(n.text or "" for n in root.iter() if _local(n) in {"t", "v", "f"}))
                    _archive_visuals(archive, result, ("ppt/media/",))
                elif suffix == ".epub":
                    for member in _epub_order(archive):
                        from .tools.web import extract_html
                        result.add(member, extract_html(_decode(archive.read(member)))[1])
                    image_members = [member for member in archive.namelist() if Path(member).suffix.lower() in IMAGE_EXTENSIONS]
                    _archive_visuals(archive, result, tuple(image_members))
                else:
                    _odf_extract(archive, suffix, result)
        elif suffix == ".rtf":
            from striprtf.striprtf import rtf_to_text
            result.add("document", rtf_to_text(raw.decode("latin-1")))
        elif suffix == ".jsonl" and '"lucifer.problem-solution.v1"' in _decode(raw).splitlines()[0]:
            from .problem_solution import FORMAT, parse_record
            seen = set()
            for line in _decode(raw).splitlines():
                if not line.strip():
                    continue
                record = parse_record(line)
                if record["id"] in seen:
                    raise ValueError("Duplicate problem/solution record id")
                seen.add(record["id"])
                # Each validated record fits one ordinary source chunk. No
                # target is split into unparseable JSON or silently omitted.
                result.add("problem/solution " + record["id"], line)
            result.metadata.update(problem_solution_format=FORMAT, examples=len(seen))
        elif suffix in {".html", ".htm"}:
            from .tools.web import extract_html
            result.add("web page", extract_html(_decode(raw))[1])
        elif suffix in {".csv", ".tsv"}:
            text = _decode(raw)
            delimiter = "\t" if suffix == ".tsv" else ","
            if suffix == ".csv":
                try:
                    delimiter = csv.Sniffer().sniff(text[:16384], delimiters=",;\t|").delimiter
                except csv.Error:
                    pass
            # The complete source stays literal, including quoted multiline cells.
            result.add("table source", text)
            result.metadata.update(delimiter=delimiter, table_format=suffix[1:])
        elif suffix in {".doc", ".ppt"}:
            if not shutil.which("libreoffice"):
                raise ValueError("Legacy Office files need LibreOffice; export to DOCX/PPTX or PDF")
            with tempfile.TemporaryDirectory(prefix="lucifer-document-") as folder:
                location = Path(folder)
                source = location / ("source" + suffix)
                source.write_bytes(raw)
                profile = location / "profile" / "user"
                profile.mkdir(parents=True)
                (profile / "registrymodifications.xcu").write_text('<?xml version="1.0"?><oor:items xmlns:oor="http://openoffice.org/2001/registry"><item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop></item></oor:items>')
                subprocess.run(["libreoffice", "-env:UserInstallation=" + (location / "profile").as_uri(),
                    "--headless", "--convert-to", "pdf", "--outdir", folder, str(source)],
                    capture_output=True, timeout=60, check=True)
                converted = location / "source.pdf"
                if not converted.is_file():
                    raise ValueError("LibreOffice could not read this document; export it to PDF")
                return extract_document("converted.pdf", converted.read_bytes(), video_frames=video_frames)
        else:
            result.add("document", _decode(raw))
    except (ValueError, ImportError):
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read {suffix or 'this document'}: {type(exc).__name__}; try an unlocked PDF or UTF-8 export") from exc
    if not result.sections and not result.visuals:
        raise ValueError("No readable text or visual content was found")
    result.metadata.update(text_bytes=result._text_bytes, visual_inputs=len(result.visuals))
    return result


def make_chunks(sections):
    chunks = []
    for section in sections:
        text = section["text"]
        for start in range(0, len(text), CHUNK_CHARS):
            value = text[start:start + CHUNK_CHARS]
            if value.strip():
                chunks.append({**section, "text": value, "offset": start, "chunk": len(chunks)})
    return chunks
