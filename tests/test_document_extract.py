"""Real parser fixtures preserve source text and identify bounded visual coverage."""
import io
import shutil
import zipfile

import pytest

from lucifer import document_extract as extraction
from lucifer.document_extract import ExtractedDocument, extract_document, make_chunks


def archive(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as output:
        for name, data in members.items():
            output.writestr(name, data)
    return stream.getvalue()


def combined(result):
    return "\n".join(section["text"] for section in result.sections)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32", "cp1252"])
def test_text_encodings_and_csv_multiline_cells_are_preserved(encoding):
    text = 'name;description\n"café";"first line\nsecond line"\n'
    result = extract_document("knowledge.csv", text.encode(encoding))
    assert combined(result) == text
    assert result.metadata["delimiter"] == ";"
    assert result.sections[0]["authority"] == "literal"


def test_binary_and_text_size_bounds_fail_explicitly_without_partial_success(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported binary"):
        extract_document("unknown.bin", b"\x00\x01binary")
    monkeypatch.setattr(extraction, "MAX_TEXT", 12)
    result = ExtractedDocument()
    result.add("first", "abc")
    with pytest.raises(ValueError, match="8 MiB"):
        result.add("too much", "é" * 6)
    assert result.sections == [{"label": "first", "text": "abc", "authority": "literal"}]


def test_html_and_rtf_become_inert_readable_text():
    html = extract_document("local.html", b"<title>Guide</title><script>EXFILTRATE</script><p>Useful <b>facts</b>.</p>")
    assert "Useful facts." in combined(html) and "EXFILTRATE" not in combined(html)
    rtf = extract_document("local.rtf", br"{\rtf1\ansi Hello \b important\b0  world.}")
    assert "Hello important world." in combined(rtf)


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF"])
def test_extensionless_images_are_detected_from_their_bytes(format):
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (24, 24), "red").save(stream, format=format)
    result = extract_document("uploaded-image", stream.getvalue())
    assert len(result.visuals) == 1 and result.sections == []


def test_word_document_preserves_paragraph_table_order_header_and_embedded_image():
    from docx import Document
    from PIL import Image
    document = Document()
    document.add_paragraph("before table")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "left cell"
    table.cell(0, 1).text = "right cell"
    document.add_paragraph("after table")
    document.sections[0].header.paragraphs[0].text = "source header"
    image = io.BytesIO()
    Image.new("RGB", (40, 20), "red").save(image, format="PNG")
    document.add_picture(io.BytesIO(image.getvalue()))
    stream = io.BytesIO()
    document.save(stream)
    result = extract_document("knowledge.docx", stream.getvalue())
    text = combined(result)
    assert text.index("before table") < text.index("left cell") < text.index("after table")
    assert "right cell" in text and "source header" in text
    assert len(result.visuals) == 1
    # Macro-enabled packages use the same inert XML path; macros are never loaded.
    renamed = extract_document("knowledge.docm", stream.getvalue())
    assert combined(renamed) == text


def test_excel_preserves_sheets_formulas_cached_values_and_source_rows():
    from openpyxl import Workbook
    book = Workbook()
    book.active.title = "Costs"
    book.active.append(["item", "amount"])
    book.active.append(["total", "=1+2"])
    book.create_sheet("Notes").append(["quoted cell", "a\nb"])
    stream = io.BytesIO()
    book.save(stream)
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as original:
        files = {name: original.read(name) for name in original.namelist()}
    files["xl/worksheets/sheet1.xml"] = files["xl/worksheets/sheet1.xml"].replace(b"<f>1+2</f><v></v>", b"<f>1+2</f><v>3</v>")
    result = extract_document("knowledge.xlsx", archive(files))
    assert [section["label"] for section in result.sections] == ["sheet Costs", "sheet Notes"]
    assert "=1+2 [cached value: 3]" in combined(result)
    assert "row 2: total" in combined(result) and "a\nb" in combined(result)


def test_odf_spreadsheet_preserves_repeated_cells_and_cached_formula_values_without_expansion():
    content = '''<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
    <office:body><office:spreadsheet><table:table table:name="Data"><table:table-row table:number-rows-repeated="2">
    <table:table-cell table:number-columns-repeated="3"><text:p>same</text:p></table:table-cell>
    <table:table-cell table:formula="of:=SUM([.A1:.C1])" office:value="6"/>
    </table:table-row><table:table-row table:number-rows-repeated="1048574"><table:table-cell table:number-columns-repeated="16384"/>
    </table:table-row></table:table></office:spreadsheet></office:body></office:document-content>'''
    result = extract_document("knowledge.ods", archive({"content.xml": content}))
    text = combined(result)
    assert "row 1..2 (repeated)" in text and "column 1..3 (repeated): same" in text
    assert "of:=SUM([.A1:.C1]) [cached value: 6]" in text
    assert len(text) < 250


def test_odf_text_preserves_inline_whitespace_without_duplicate_table_text():
    content = '''<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>
    <text:p>Hello<text:s/><text:span>world</text:span><text:line-break/>next</text:p>
    </office:text></office:body></office:document-content>'''
    result = extract_document("knowledge.odt", archive({"content.xml": content}))
    assert combined(result) == "Hello world\nnext"


def test_epub_follows_manifest_spine_reading_order_not_zip_order():
    container = '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>'
    package = '<package><manifest><item id="first" href="first.xhtml"/><item id="second" href="second.xhtml"/></manifest><spine><itemref idref="first"/><itemref idref="second"/></spine></package>'
    result = extract_document("knowledge.epub", archive({
        "OPS/second.xhtml": "<html><p>second chapter</p></html>",
        "OPS/first.xhtml": "<html><p>first chapter</p></html>",
        "META-INF/container.xml": container, "OPS/book.opf": package,
    }))
    assert [section["label"] for section in result.sections] == ["OPS/first.xhtml", "OPS/second.xhtml"]
    assert combined(result).index("first chapter") < combined(result).index("second chapter")


def test_presentation_follows_deck_order_and_preserves_notes_and_chart_values():
    presentation = '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="10" r:id="r10"/><p:sldId id="2" r:id="r2"/></p:sldIdLst></p:presentation>'
    relations = '<Relationships><Relationship Id="r10" Target="slides/slide10.xml"/><Relationship Id="r2" Target="slides/slide2.xml"/></Relationships>'
    result = extract_document("knowledge.pptx", archive({
        "ppt/presentation.xml": presentation, "ppt/_rels/presentation.xml.rels": relations,
        "ppt/slides/slide2.xml": '<slide xmlns:a="a"><a:t>second slide</a:t></slide>',
        "ppt/slides/slide10.xml": '<slide xmlns:a="a"><a:t>first slide</a:t></slide>',
        "ppt/notesSlides/notesSlide10.xml": '<notes xmlns:a="a"><a:t>speaker notes</a:t></notes>',
        "ppt/charts/chart1.xml": '<chart xmlns:c="c"><c:v>42</c:v></chart>',
    }))
    labels = [section["label"] for section in result.sections]
    assert labels.index("ppt/slides/slide10.xml") < labels.index("ppt/slides/slide2.xml")
    assert "speaker notes" in combined(result) and "42" in combined(result)


def readable_pdf():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 10 100 Td (Readable PDF facts) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_pdf_text_is_literal_and_scanned_page_is_available_for_loaded_vision():
    result = extract_document("facts.pdf", readable_pdf())
    assert "Readable PDF facts" in combined(result) and result.sections[0]["label"] == "page 1"
    if not shutil.which("pdftoppm"):
        pytest.skip("pdftoppm unavailable")
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (40, 40), "red").save(stream, format="PDF")
    scanned = extract_document("scan.pdf", stream.getvalue())
    assert scanned.sections == [] and len(scanned.visuals) == 1
    assert scanned.metadata["pages"] == 1
    with Image.open(io.BytesIO(scanned.visuals[0][1])) as image:
        assert image.width <= 1536 and image.height <= 1536


def test_short_real_video_samples_first_and_last_display_intervals_with_honest_audio_warning():
    import av
    from PIL import Image
    stream = io.BytesIO()
    with av.open(stream, "w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=2)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        for number in range(4):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (number * 70, 20, 30)))
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
    result = extract_document("clip.mp4", stream.getvalue(), video_frames=8)
    assert result.metadata["duration_seconds"] == 2.0
    assert result.metadata["sampled_frames"] == 4
    assert result.visuals[0][0] == "frame at 0.000s"
    assert result.visuals[-1][0] == "frame at 1.500s"
    assert any("Audio is not transcribed" in warning for warning in result.warnings)


def test_real_embedded_text_subtitles_become_literal_timestamped_knowledge():
    import av
    from PIL import Image
    source = av.open(io.BytesIO(b"1\n00:00:00,000 --> 00:00:01,000\nHello knowledge\n"), format="srt")
    output = io.BytesIO()
    try:
        with av.open(output, "w", format="matroska") as container:
            video = container.add_stream("mpeg4", rate=2)
            video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
            subtitle = container.add_stream_from_template(source.streams[0])
            for number in range(4):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (number * 70, 20, 30)))
                for packet in video.encode(frame):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            for packet in source.demux():
                if packet.dts is not None:
                    packet.stream = subtitle
                    container.mux(packet)
    finally:
        source.close()
    result = extract_document("subtitled.mkv", output.getvalue(), video_frames=2)
    assert "[0.000s] Hello knowledge" in combined(result)
    assert result.sections[0]["authority"] == "literal"


def test_video_extension_cannot_open_an_external_playlist(tmp_path):
    private = tmp_path / "private.png"
    from PIL import Image
    Image.new("RGB", (20, 20), "red").save(private)
    playlist = f"#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\n{private}\n#EXT-X-ENDLIST\n"
    with pytest.raises(ValueError, match="Invalid data|Cannot read"):
        extract_document("disguised.mp4", playlist.encode())


def test_zip_bomb_bounds_and_xml_entities_are_rejected(monkeypatch):
    malformed = archive({"word/document.xml": '<!DOCTYPE doc [<!ENTITY secret "private">]><doc><p>&secret;</p></doc>'})
    with pytest.raises(ValueError, match="EntitiesForbidden|Cannot read"):
        extract_document("document.docx", malformed)
    data = archive({"word/document.xml": "x" * 100})
    monkeypatch.setattr(extraction, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="128 MiB"):
        extract_document("document.docx", data)


def test_knowledge_chunks_preserve_all_characters_with_source_offsets():
    text = "before\n" + "facts " * 1000 + "\nlast fact"
    sections = [{"label": "page 2", "text": text, "authority": "literal"}]
    chunks = make_chunks(sections)
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert [chunk["offset"] for chunk in chunks] == list(range(0, len(text), extraction.CHUNK_CHARS))
    assert all(chunk["label"] == "page 2" for chunk in chunks)
