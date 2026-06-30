from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from PIL import Image, ImageDraw, ImageFont


BASE = Path(r"C:\steganalysis_project")
OUT_DIR = BASE / "outputs" / "stegguard_report_revision" / "final"
IMG_DIR = OUT_DIR / "prepared_images"
DOWNLOADS = Path(r"C:\Users\lizag\Downloads")
EXTRACTED = BASE / "outputs" / "stegguard_report_revision" / "diagram_doc_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

OUT_DOCX = OUT_DIR / "StegGuard_SOC_Revised_Interim_Report.docx"


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def font(name: str = "Arial", size: int = 18, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int):
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = (line + " " + word).strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def arrow(draw: ImageDraw.ImageDraw, p1, p2, fill=(70, 70, 70), width=3):
    draw.line([p1, p2], fill=fill, width=width)
    x1, y1 = p1
    x2, y2 = p2
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 11
    pts = [
        (x2, y2),
        (x2 - head * math.cos(ang - math.pi / 6), y2 - head * math.sin(ang - math.pi / 6)),
        (x2 - head * math.cos(ang + math.pi / 6), y2 - head * math.sin(ang + math.pi / 6)),
    ]
    draw.polygon(pts, fill=fill)


def draw_center_text(draw, box, text, fnt, fill=(45, 45, 45)):
    x, y, w, h = box
    lines = wrap(draw, text, fnt, max(20, w - 18))
    line_h = fnt.size + 3 if hasattr(fnt, "size") else 16
    total_h = line_h * len(lines)
    cy = y + (h - total_h) / 2
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=fnt)
        draw.text((x + (w - (bb[2] - bb[0])) / 2, cy), line, font=fnt, fill=fill)
        cy += line_h


def create_existing_workflow() -> Path:
    path = IMG_DIR / "existing_workflow_generated.png"
    img = Image.new("RGB", (1000, 1450), "white")
    d = ImageDraw.Draw(img)
    f = font(size=20)
    f_small = font(size=16)
    d.rectangle((70, 30, 930, 1410), outline=(160, 160, 160), width=3)
    d.text((340, 45), "Swimlane: Security Analyst", font=font(size=20, bold=True), fill=(45, 45, 45))
    nodes = {
        "start": ("oval", 445, 95, 110, 45, "Start"),
        "recv": ("para", 340, 175, 320, 55, "Email attachment received"),
        "open": ("rect", 365, 270, 270, 60, "Open email manually"),
        "download": ("rect", 360, 370, 280, 60, "Download attachment"),
        "type": ("diamond", 405, 485, 190, 140, "File type identified?"),
        "img": ("rect", 120, 700, 230, 58, "Inspect image manually"),
        "aud": ("rect", 385, 700, 230, 58, "Inspect audio manually"),
        "doc": ("rect", 650, 700, 230, 58, "Inspect document or other file"),
        "tool": ("double", 335, 830, 330, 62, "Run separate manual analysis tool"),
        "sus": ("diamond", 400, 970, 200, 145, "Suspicious indicators found?"),
        "esc": ("rect", 165, 1190, 240, 62, "Escalate to SOC team"),
        "ben": ("rect", 595, 1190, 240, 62, "Mark file as benign"),
        "report": ("para", 360, 1290, 280, 60, "Manual findings report"),
        "end": ("oval", 445, 1380, 110, 45, "End"),
    }
    def draw_node(key):
        shape, x, y, w, h, label = nodes[key]
        if shape == "rect":
            d.rectangle((x, y, x + w, y + h), outline=(120, 120, 120), fill=(240, 240, 240), width=3)
        elif shape == "double":
            d.rectangle((x, y, x + w, y + h), outline=(120, 120, 120), fill=(240, 240, 240), width=3)
            d.line((x + 10, y, x + 10, y + h), fill=(120, 120, 120), width=3)
            d.line((x + w - 10, y, x + w - 10, y + h), fill=(120, 120, 120), width=3)
        elif shape == "oval":
            d.rounded_rectangle((x, y, x + w, y + h), radius=22, outline=(120, 120, 120), fill=(240, 240, 240), width=3)
        elif shape == "para":
            d.polygon([(x + 25, y), (x + w, y), (x + w - 25, y + h), (x, y + h)], outline=(120, 120, 120), fill=(240, 240, 240))
            d.line([(x + 25, y), (x + w, y), (x + w - 25, y + h), (x, y + h), (x + 25, y)], fill=(120, 120, 120), width=3)
        elif shape == "diamond":
            pts = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]
            d.polygon(pts, outline=(120, 120, 120), fill=(240, 240, 240))
            d.line(pts + [pts[0]], fill=(120, 120, 120), width=3)
        draw_center_text(d, (x, y, w, h), label, f_small)
    for k in nodes:
        draw_node(k)
    def bottom(k):
        _, x, y, w, h, _ = nodes[k]
        return (x + w / 2, y + h)
    def top(k):
        _, x, y, w, h, _ = nodes[k]
        return (x + w / 2, y)
    for a, b in [("start", "recv"), ("recv", "open"), ("open", "download"), ("download", "type")]:
        arrow(d, bottom(a), top(b))
    arrow(d, (405, 555), (235, 700)); d.text((250, 645), "Image", font=f_small, fill=(30, 30, 30))
    arrow(d, (500, 625), (500, 700)); d.text((512, 650), "Audio", font=f_small, fill=(30, 30, 30))
    arrow(d, (595, 555), (765, 700)); d.text((700, 645), "Document/Other", font=f_small, fill=(30, 30, 30))
    for k in ["img", "aud", "doc"]:
        arrow(d, bottom(k), top("tool"))
    arrow(d, bottom("tool"), top("sus"))
    arrow(d, (400, 1040), (285, 1190)); d.text((285, 1105), "Yes", font=f_small, fill=(30, 30, 30))
    arrow(d, (600, 1040), (715, 1190)); d.text((695, 1105), "No", font=f_small, fill=(30, 30, 30))
    arrow(d, bottom("esc"), top("report"))
    arrow(d, bottom("ben"), top("report"))
    arrow(d, bottom("report"), top("end"))
    img.save(path, quality=95)
    return path


def create_usecase_diagram() -> Path:
    path = IMG_DIR / "usecase_generated.png"
    img = Image.new("RGB", (1300, 820), "white")
    d = ImageDraw.Draw(img)
    f = font(size=18)
    fb = font(size=20, bold=True)
    d.rectangle((330, 70, 1230, 760), outline=(120, 120, 120), width=3)
    d.text((760, 85), "StegGuard SOC System", font=fb, fill=(40, 40, 40), anchor="mm")
    actors = {
        "Security Analyst": (125, 180),
        "System Administrator": (125, 390),
        "Compliance Auditor": (125, 610),
    }
    def actor(name, pos):
        x, y = pos
        d.ellipse((x - 14, y - 55, x + 14, y - 27), outline=(80, 80, 80), width=2)
        d.line((x, y - 27, x, y + 30), fill=(80, 80, 80), width=2)
        d.line((x - 35, y - 5, x + 35, y - 5), fill=(80, 80, 80), width=2)
        d.line((x, y + 30, x - 30, y + 70), fill=(80, 80, 80), width=2)
        d.line((x, y + 30, x + 30, y + 70), fill=(80, 80, 80), width=2)
        d.text((x, y + 88), name, font=f, fill=(40, 40, 40), anchor="mm")
    for name, pos in actors.items():
        actor(name, pos)
    ucs = {
        "Upload file": (500, 155, 220, 70),
        "Run steganalysis": (795, 155, 260, 70),
        "View result": (1080, 155, 210, 70),
        "Upload batch": (500, 310, 220, 70),
        "View dashboard": (795, 310, 260, 70),
        "Export report": (1080, 310, 210, 70),
        "Configure thresholds": (500, 500, 260, 70),
        "Manage users": (795, 500, 220, 70),
        "Retrieve audit logs": (1080, 500, 240, 70),
        "Record audit event": (795, 660, 270, 70),
    }
    for label, (x, y, w, h) in ucs.items():
        d.ellipse((x - w/2, y - h/2, x + w/2, y + h/2), outline=(120, 120, 120), fill=(245, 245, 245), width=3)
        draw_center_text(d, (x - w/2, y - h/2, w, h), label, f)
    def line_actor(actor_name, uc_name):
        ax, ay = actors[actor_name]
        x, y, w, h = ucs[uc_name]
        d.line((ax + 40, ay, x - w/2, y), fill=(80, 80, 80), width=2)
    for uc in ["Upload file", "Run steganalysis", "View result", "Upload batch", "View dashboard", "Export report"]:
        line_actor("Security Analyst", uc)
    for uc in ["View dashboard", "Configure thresholds", "Manage users"]:
        line_actor("System Administrator", uc)
    for uc in ["Retrieve audit logs", "Export report"]:
        line_actor("Compliance Auditor", uc)
    def dashed(a, b, text):
        x1, y1, w1, h1 = ucs[a]
        x2, y2, w2, h2 = ucs[b]
        # dashed line between use cases
        x1e = x1 + w1/2
        x2e = x2 - w2/2
        y = (y1 + y2) / 2
        cur = x1e
        while cur < x2e - 12:
            d.line((cur, y, min(cur + 14, x2e), y), fill=(80, 80, 80), width=2)
            cur += 24
        d.polygon([(x2e, y), (x2e - 12, y - 6), (x2e - 12, y + 6)], fill=(80, 80, 80))
        d.text(((x1e + x2e) / 2, y - 24), text, font=font(size=14), fill=(40, 40, 40), anchor="mm")
    dashed("Upload file", "Run steganalysis", "<<include>>")
    dashed("Run steganalysis", "View result", "<<include>>")
    dashed("View result", "Export report", "<<extend>>")
    dashed("Run steganalysis", "Record audit event", "<<include>>")
    dashed("Retrieve audit logs", "Record audit event", "<<include>>")
    img.save(path, quality=95)
    return path


def flatten_image(src: Path, name: str) -> Path:
    out = IMG_DIR / name
    im = Image.open(src)
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, "white")
        bg.alpha_composite(rgba)
        im = bg.convert("RGB")
    else:
        im = im.convert("RGB")
    im.save(out, quality=95)
    return out


main_words = []


def track(text: str):
    main_words.append(text)


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_col_width(cell, width_inches):
    cell.width = Inches(width_inches)
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def configure_styles(doc: Document):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.font.size = Pt(11)
    pf = normal.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.10

    for style_name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 16, 8),
        ("Heading 2", 13, "2E74B5", 12, 6),
        ("Heading 3", 12, "1F4D78", 8, 4),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    if "Caption" in styles:
        cap = styles["Caption"]
        cap.font.name = "Calibri"
        cap.font.size = Pt(9)
        cap.font.italic = True
        cap.font.color.rgb = RGBColor(80, 80, 80)
        cap.paragraph_format.space_before = Pt(3)
        cap.paragraph_format.space_after = Pt(8)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.add_run("Page ")
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_text = OxmlElement("w:t")
    fld_text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_text)
    run._r.append(fld_end)


def set_section_page_start(section, start=1):
    sect_pr = section._sectPr
    pg_num_type = sect_pr.find(qn("w:pgNumType"))
    if pg_num_type is None:
        pg_num_type = OxmlElement("w:pgNumType")
        sect_pr.append(pg_num_type)
    pg_num_type.set(qn("w:start"), str(start))


def set_update_fields(doc):
    settings = doc.settings.element
    existing = settings.find(qn("w:updateFields"))
    if existing is None:
        existing = OxmlElement("w:updateFields")
        settings.append(existing)
    existing.set(qn("w:val"), "true")


def add_toc(paragraph):
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), r'TOC \o "1-3" \h \z \u')
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "Table of contents will update automatically in Word."
    run.append(text)
    fld.append(run)
    paragraph._p.append(fld)


def add_heading(doc, text: str, level: int):
    p = doc.add_heading(text, level=level)
    track(text)
    return p


def add_para(doc, text: str, *, bold_label: str | None = None, main=True, align=None):
    p = doc.add_paragraph()
    if bold_label:
        r = p.add_run(bold_label)
        r.bold = True
        p.add_run(text)
        full = bold_label + text
    else:
        p.add_run(text)
        full = text
    if align is not None:
        p.alignment = align
    if main:
        track(full)
    return p


def add_bullets(doc, items, main=True):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(4)
        p.add_run(item)
        if main:
            track(item)


def add_numbered(doc, items, main=True):
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(4)
        p.add_run(item)
        if main:
            track(item)


def add_caption(doc, text: str, main=True):
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text)
    if main:
        track(text)


def add_table(doc, caption: str, headers, rows, widths=None, main=True):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        set_cell_shading(hdr[i], "F2F4F7")
        set_cell_margins(hdr[i])
        hdr[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for r in hdr[i].paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cells[i])
    if widths:
        for row in table.rows:
            for idx, width in enumerate(widths):
                set_col_width(row.cells[idx], width)
    add_caption(doc, caption, main=main)
    if main:
        track(" ".join(headers))
        for row in rows:
            track(" ".join(row))
    return table


def add_figure(doc, src: Path, caption: str, *, max_width=6.25, max_height=8.25, main=True):
    if not src.exists():
        add_para(doc, f"[Missing figure file: {src}]", main=main)
        return
    prepared = flatten_image(src, src.stem.replace(" ", "_").replace("(", "").replace(")", "") + "_white.jpg")
    im = Image.open(prepared)
    w, h = im.size
    ratio = w / h
    draw_w = max_width
    draw_h = draw_w / ratio
    if draw_h > max_height:
        draw_h = max_height
        draw_w = draw_h * ratio
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(prepared), width=Inches(draw_w), height=Inches(draw_h))
    add_caption(doc, caption, main=main)


def add_source_note(doc, text: str, main=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(90, 90, 90)
    if main:
        track(text)


def build_doc():
    doc = Document()
    configure_styles(doc)
    set_update_fields(doc)

    # Front matter without page numbering.
    cover = doc.sections[0]
    cover.footer.is_linked_to_previous = False
    cover.footer.paragraphs[0].text = ""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("\n\nA DISSERTATION INTERIM REPORT\n")
    r.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor.from_string("0B2545")
    p.add_run("\nON\n").bold = True
    title = p.add_run("\nMULTI-MODAL STEGANALYSIS FOR CORPORATE PHISHING DEFENSE\n")
    title.bold = True
    title.font.size = Pt(17)
    title.font.color.rgb = RGBColor.from_string("0B2545")
    p.add_run("\nStegGuard SOC System\n\n").bold = True
    details = [
        "Student Name: Liza Gurung",
        "Student ID: 24812928",
        "Programme: BSc. Computing",
        "Supervisor: Mr. Nischal Khadka",
        "Institution: Naaya Aayam Multi-Disciplinary Institute",
        "Awarding Body: University of Northampton",
        "Academic Year: 2026",
        "Word Count: approximately 5,500 words excluding appendices",
    ]
    for line in details:
        pp = doc.add_paragraph(line)
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    toc_title = doc.add_paragraph()
    toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = toc_title.add_run("Table of Contents")
    rr.bold = True
    rr.font.size = Pt(16)
    rr.font.color.rgb = RGBColor.from_string("0B2545")
    add_toc(doc.add_paragraph())
    doc.add_page_break()

    main_section = doc.add_section(WD_SECTION.NEW_PAGE)
    main_section.top_margin = Inches(1)
    main_section.bottom_margin = Inches(1)
    main_section.left_margin = Inches(1)
    main_section.right_margin = Inches(1)
    main_section.footer.is_linked_to_previous = False
    set_section_page_start(main_section, 1)
    add_page_number(main_section.footer.paragraphs[0])

    existing_workflow = create_existing_workflow()
    usecase = create_usecase_diagram()

    # Chapter 1
    add_heading(doc, "1. Introduction", 1)
    add_para(doc, "Corporate Security Operations Centres (SOCs) now receive a large volume of files through email, collaboration platforms and ticketing channels. Attackers increasingly exploit this file flow because attachments are trusted by business users and are processed quickly by analysts under operational pressure. Steganography is especially difficult in this setting because the harmful content is hidden inside an apparently normal carrier, such as a PNG image or WAV audio clip. The StegGuard SOC project responds to this risk by designing and implementing a multi-modal steganalysis system for corporate phishing defence.")
    add_para(doc, "The project is framed as an applied computing dissertation. It combines requirements engineering, artefact design, algorithm implementation and evaluation into one practical system. The system accepts individual or batch file uploads, validates file type and size, extracts image and audio features, applies a detection model, records audit information and presents results through a SOC-style dashboard. The revised report therefore moves beyond a purely algorithmic description and treats StegGuard as an end-to-end security workflow.")
    add_heading(doc, "1.1 Background", 2)
    add_para(doc, "Steganography differs from encryption because it hides the existence of communication rather than only hiding the meaning of the message. A carrier file may open normally and appear harmless while containing manipulated pixel values, audio samples or metadata. In phishing contexts this property is useful to attackers because a malicious instruction, staging marker or payload fragment may bypass conventional signature checks. Defensive teams need tools that can inspect the statistical structure of media files rather than relying only on known malware hashes.")
    add_para(doc, "Image steganalysis often uses residual noise patterns, co-occurrence statistics and rich feature models to identify unnatural changes in the image signal. Audio steganalysis can use spectral and cepstral representations, because hidden data can disturb the distribution of frequency content across small time windows. StegGuard uses Spatial Rich Model (SRM) ideas for image feature extraction and Mel-Frequency Cepstral Coefficient (MFCC) features for audio. These two modalities are then fused so that the SOC workflow can handle both image and audio attachments through one interface.")
    add_heading(doc, "1.2 Rationale", 2)
    add_para(doc, "The rationale for the project is that manual attachment analysis is slow, fragmented and inconsistent. An analyst may need to download a suspicious file, identify its type, choose a separate tool, interpret unfamiliar output and then create a report. That process is not realistic when a SOC receives many alerts in a short period. Manual analysis also creates weak auditability because notes may be stored separately from the file metadata, threshold settings and final verdict.")
    add_para(doc, "StegGuard addresses this operational gap by automating repetitive analysis steps while still leaving review decisions visible to the analyst. The system is designed for files commonly encountered in phishing investigations, especially PNG images and WAV audio. Its value is not only in the classifier but also in the surrounding controls: validation, feature extraction, result storage, dashboard statistics, batch processing, threshold configuration and audit logging. This makes the project suitable as a dissertation artefact because it demonstrates both technical detection logic and responsible system design.")
    add_heading(doc, "1.3 Aim and Objectives", 2)
    add_para(doc, "To design, implement and evaluate StegGuard SOC, a multi-modal steganalysis system that supports corporate phishing defence by detecting suspicious image and audio attachments, recording auditable results and presenting analyst-friendly dashboard feedback.", bold_label="Aim: ")
    add_bullets(doc, [
        "Analyse the problem domain of manual email attachment inspection and identify weaknesses in speed, consistency, auditability and multi-modal coverage.",
        "Review comparable steganalysis tools and research approaches, including StegExpose, Stegdetect and SteganoGAN, to position the proposed solution.",
        "Design system workflows, UML diagrams, database relationships and interface wireframes that reflect the expected SOC user journey.",
        "Implement SRM-based image feature extraction and MFCC-based audio feature extraction as reusable parts of the detection pipeline.",
        "Develop a prototype SOC web application with file upload, batch processing, dashboard statistics, threshold configuration and audit log retrieval.",
        "Evaluate the prototype using algorithm tests, integration tests, functional scenarios and visual testing evidence captured from the implemented system.",
    ])
    add_heading(doc, "1.4 Overview of Methodology", 2)
    add_para(doc, "The dissertation follows an applied design and development methodology. The research begins with a literature and comparable systems review, then converts the observed gap into system requirements. The artefact is developed using an Agile Scrum approach because the project contains several dependent increments: data preparation, feature extraction, API development, dashboard design, batch processing and testing. Each sprint produces a small working increment that can be reviewed against the dissertation objectives.")
    add_para(doc, "Scrum is used in a lightweight student-project form. The product backlog contains user stories such as upload a file, validate supported formats, extract SRM features, classify audio files, save audit logs and refresh dashboard metrics. Sprint planning selects a narrow set of stories, implementation produces a testable increment and sprint review checks whether the increment contributes to the research aim. This approach is suitable because uncertainty exists around model behaviour, interface clarity and testing evidence; short cycles allow the design to be refined without losing traceability.")

    # Chapter 2
    add_heading(doc, "2. Requirements Engineering", 1)
    add_para(doc, "Requirements engineering is used to translate the security problem into a system that can be built and evaluated. The chapter reviews literature, compares existing tools, defines the research and development methodology, and states functional and non-functional requirements. It also clarifies performance expectations and the role of the implemented algorithms. The emphasis is on the whole SOC workflow, not only the classifier.")
    add_heading(doc, "2.1 Literature Review", 2)
    add_para(doc, "The literature on steganalysis shows a movement from handcrafted statistical features toward learning-based and hybrid approaches. Classical image steganalysis uses residual filters and co-occurrence matrices to detect weak embedding artefacts. The Spatial Rich Model is relevant because it represents many residual patterns and remains interpretable as a feature extraction method. Research by Bas, Filler and Pevny (2011) and by Kodovsky, Fridrich and Holub (2012) provides a foundation for feature-rich image steganalysis and ensemble classification.")
    add_para(doc, "Deep learning approaches can improve detection where the relationship between embedding artefacts and carrier statistics is complex. CNN layers are useful for local spatial patterns, while recurrent units can model sequential or fused feature behaviour. In audio analysis, MFCC features are widely used because they summarise the spectral envelope of a signal in a compact form. For steganalysis, MFCC values may reveal disruptions introduced by sample-level embedding. This dissertation adopts a pragmatic hybrid approach: established feature extraction methods are combined with a classifier pipeline and wrapped in an operational SOC application.")
    add_heading(doc, "2.2 Comparable Systems", 2)
    add_para(doc, "Comparable systems were selected because they represent different approaches to steganography detection and generation. StegExpose focuses on classical image LSB detection, Stegdetect targets older JPEG steganography tools and SteganoGAN demonstrates a modern generative approach. They are useful references, but none directly provides the integrated multi-modal SOC workflow required by this project.")
    add_heading(doc, "2.2.1 StegExpose", 3)
    add_para(doc, "StegExpose is an image steganalysis tool that combines several statistical detectors for least significant bit embedding. It is useful for research and forensic triage because it gives a direct way to test image carriers. Its limitation for this project is that it is primarily image-oriented and does not provide dashboard, audit, batch workflow or audio support.")
    add_heading(doc, "2.2.2 Stegdetect", 3)
    add_para(doc, "Stegdetect was designed to detect steganographic content produced by older JPEG steganography programs. It is historically important because it shows how targeted signatures can identify known embedding tools. However, its narrow file coverage and age make it less suitable for current corporate workflows where attackers may use PNG images, WAV audio or custom scripts.")
    add_heading(doc, "2.2.3 SteganoGAN", 3)
    add_para(doc, "SteganoGAN uses generative adversarial network concepts to hide and recover messages in images. It is relevant because it demonstrates how machine learning can change the steganography threat landscape. For this dissertation it is treated as a comparable research system rather than an operational SOC product, because the project focuses on detection, auditability and analyst workflows.")
    add_table(doc, "Table 2.1: Comparable system analysis. Source: Author generated, 2026.",
              ["System", "Main focus", "Strength", "Limitation for StegGuard"],
              [
                  ["StegExpose", "Image LSB steganalysis", "Combines classical statistical detectors", "No audio support, SOC dashboard or audit workflow"],
                  ["Stegdetect", "JPEG steganography detection", "Targets known historical tools", "Limited modern file coverage and limited integration"],
                  ["SteganoGAN", "Learning-based steganography", "Shows modern adversarial embedding potential", "Focused on generation rather than SOC detection"],
                  ["StegGuard SOC", "Multi-modal SOC detection", "Combines SRM, MFCC, dashboard and audit logging", "Prototype requires further dataset expansion and deployment testing"],
              ],
              widths=[1.2, 1.5, 1.8, 2.0])
    add_heading(doc, "2.3 Research Methodology", 2)
    add_para(doc, "The dissertation uses a design science research methodology. The central research output is an artefact: a working StegGuard SOC prototype supported by diagrams, requirements, algorithms and tests. The methodology is appropriate because the research question is practical: how can a SOC analyst inspect suspicious image and audio attachments more consistently? The artefact is evaluated through scenario execution, test results and comparison against the stated requirements.")
    add_para(doc, "The methodology contains four linked activities. First, the problem is investigated through literature and comparable systems. Second, requirements are specified for users, system actions, data storage, algorithms and testing. Third, the system is designed and implemented through Agile increments. Fourth, the prototype is evaluated using test cases, screenshots and performance observations. The method therefore connects academic review with practical engineering.")
    add_heading(doc, "2.4 Research Philosophy", 2)
    add_para(doc, "The research philosophy is pragmatic. The project does not attempt to prove that one philosophical view of security is universally correct. Instead, it asks what works for a defined operational problem and how the result can be justified through evidence. Pragmatism fits the project because successful steganalysis in a SOC depends on measurable detection behaviour, usable interfaces, audit records and clear analyst decisions.")
    add_heading(doc, "2.5 Development Methodology", 2)
    add_para(doc, "The development methodology is Agile Scrum adapted for an individual dissertation. The product backlog is divided into sprints that mirror the system architecture. Early sprints focus on problem definition, literature review and workflow design. Middle sprints produce SRM and MFCC extraction modules, API endpoints and database storage. Later sprints implement dashboard statistics, batch processing, audit log retrieval, threshold settings and testing evidence.")
    add_para(doc, "Each sprint follows a simple Scrum cycle. Sprint planning selects a small set of features, implementation creates the increment, testing checks whether the increment works, and sprint review updates the backlog. For example, the batch upload sprint produced file queue behaviour, validation rules, progress feedback, summary metrics and audit entries. This iterative approach reduces risk because the project can be evaluated as a working system even before every final improvement is complete.")
    add_para(doc, "The Scrum artefacts are kept lightweight but explicit. The product backlog is represented by requirements and user stories; the sprint backlog is represented by the set of functions selected for implementation during a short period; the increment is represented by a tested screen, endpoint, diagram or algorithm module. Definition of done requires more than code completion. A feature is treated as done only when it has visible output, controlled error handling, a relevant diagram or design note, and at least one form of verification. This helps the dissertation remain traceable because each implemented function can be connected back to requirements and forward to testing evidence.")
    add_heading(doc, "2.6 Evaluation Strategy", 2)
    add_para(doc, "The evaluation strategy uses more than one type of evidence. Algorithm tests check whether feature extraction returns the expected structure and whether probability-to-verdict logic behaves at boundary values. Integration tests check whether valid image and audio files return successful responses and whether unsupported or corrupt files return controlled errors. Scenario tests check user-visible workflows such as single file analysis, batch processing, audit log retrieval and dashboard refresh.")
    add_para(doc, "The evaluation also considers non-functional criteria. Processing time, usability, auditability, error handling and security controls are examined because a SOC tool must be dependable under repeated use. The project does not claim production certification, but it provides structured evidence that the prototype meets the dissertation requirements and identifies future work for deployment-level assurance.")
    add_heading(doc, "2.7 Methodology for Comparable Analysis", 2)
    add_para(doc, "The comparable analysis uses four criteria: file modality, detection method, operational integration and evaluability. File modality asks whether the tool supports images, audio or both. Detection method identifies whether it uses classical statistics, signatures, deep learning or generative methods. Operational integration checks whether the tool provides dashboard, audit, reports and batch workflows. Evaluability checks whether the tool can support clear tests and measurable outputs. StegGuard is positioned against these criteria rather than only against raw detection accuracy.")
    add_heading(doc, "2.8 Requirement Specification", 2)
    add_heading(doc, "2.8.1 Problem Domain", 3)
    add_para(doc, "The problem domain is corporate phishing defence, where analysts must inspect suspicious attachments without delaying legitimate business activity. Files may be benign, suspicious, corrupt or unsupported. The system must therefore validate input before analysis, classify supported files, reject unsupported formats safely and keep a record of what happened. The domain also requires accountability because decisions may be reviewed during incident response or compliance checks.")
    add_heading(doc, "2.8.2 Proposed Solution Strategy", 3)
    add_para(doc, "The proposed solution is a web-based SOC prototype that automates multi-modal steganalysis while preserving analyst control. A user uploads a single file or batch of files. The system validates file type and size, determines modality, extracts SRM or MFCC features, normalises the feature vector, runs the classifier, stores the result, records an audit entry and displays the verdict. Dashboard metrics summarise recent activity so the analyst can monitor detections, false positive rate and processing time.")
    add_heading(doc, "2.8.3 Alternative Solution Strategy", 3)
    add_para(doc, "Alternative approaches were considered but were not selected as the main strategy:")
    add_bullets(doc, [
        "Use separate command-line tools for image and audio analysis. This is simple to prototype but weak for SOC auditability, usability and batch processing.",
        "Build only a deep learning classifier without a dashboard. This may improve modelling focus but would not address the operational workflow required by analysts.",
        "Use only metadata and signature checks. This is fast, but it would miss hidden payloads that do not change obvious metadata or match known signatures.",
        "Outsource file analysis to a cloud API. This could reduce local build complexity, but it raises privacy, cost and data handling concerns for corporate attachments.",
    ])
    add_heading(doc, "2.8.4 Functional Requirements", 3)
    add_table(doc, "Table 2.2: Functional requirements. Source: Author generated, 2026.",
              ["ID", "Requirement", "Priority", "Verification"],
              [
                  ["FR1", "Allow analysts to upload a single PNG or WAV file for analysis.", "High", "Single file scenario and API test"],
                  ["FR2", "Validate file type, size and format before classification.", "High", "Invalid file and corrupt file tests"],
                  ["FR3", "Extract SRM features for supported image files.", "High", "Algorithm unit tests"],
                  ["FR4", "Extract MFCC features for supported audio files.", "High", "Algorithm and integration tests"],
                  ["FR5", "Classify each file as clean, review or suspicious using a confidence score.", "High", "Boundary probability tests"],
                  ["FR6", "Support batch upload and queued processing for multiple files.", "Medium", "Batch scenario test"],
                  ["FR7", "Store detection result, file metadata and audit event records.", "High", "Database and log retrieval tests"],
                  ["FR8", "Refresh dashboard statistics for detections, throughput and model status.", "Medium", "Dashboard refresh scenario"],
                  ["FR9", "Allow an administrator to configure detection thresholds.", "Medium", "Threshold activity scenario"],
                  ["FR10", "Export or prepare report evidence for analyst review.", "Medium", "Report workflow inspection"],
              ],
              widths=[0.6, 3.2, 0.8, 1.9])
    add_heading(doc, "2.8.5 Non-Functional Requirements", 3)
    add_table(doc, "Table 2.3: Non-functional requirements. Source: Author generated, 2026.",
              ["ID", "Quality attribute", "Requirement", "Verification"],
              [
                  ["NFR1", "Performance", "Return typical single-file responses within a few seconds on the development machine.", "Processing time observation"],
                  ["NFR2", "Security", "Hash files before storage and avoid storing raw unsupported file content.", "Code and UI evidence"],
                  ["NFR3", "Reliability", "Handle invalid, corrupt and unsupported files without crashing.", "Negative test cases"],
                  ["NFR4", "Usability", "Provide a dashboard, status labels and clear verdicts for analyst interpretation.", "Interface screenshots"],
                  ["NFR5", "Auditability", "Record upload, classification, error and threshold events with timestamps.", "Audit log scenario"],
                  ["NFR6", "Maintainability", "Separate validation, extraction, classification, API and logging modules.", "Class and sequence diagrams"],
                  ["NFR7", "Testability", "Provide deterministic unit and integration tests for key algorithm and API behaviours.", "Pytest evidence"],
                  ["NFR8", "Scalability", "Support batch workflow and future queue/worker scaling.", "Batch processing design"],
              ],
              widths=[0.6, 1.2, 3.1, 1.6])
    add_heading(doc, "2.8.6 Goals of Implementation", 3)
    add_para(doc, "The implementation goals are to produce a working prototype, not a theoretical design only. The prototype should demonstrate file validation, feature extraction, classification, result storage, audit logging, dashboard presentation and tests. A secondary goal is to make the design understandable through diagrams so that the dissertation reader can follow how the system behaves before reading implementation details.")
    add_para(doc, "A further goal is traceability across chapters. Chapter 2 defines what the system must do, Chapter 3 shows how those requirements are designed, Chapter 4 explains how the prototype is built and Chapter 5 verifies the behaviour. This structure is important because dissertation assessment is not based only on whether a screen exists. It also depends on whether the research problem, method, design decisions and evaluation evidence are logically connected.")
    add_heading(doc, "2.8.7 System Workflow Diagrams", 3)
    add_para(doc, "The existing workflow shows why automation is required. Manual attachment analysis forces the analyst to move between email, local storage, separate analysis tools, notes and escalation. The proposed workflow centralises these activities inside StegGuard so that validation, modality detection, feature extraction, classification and logging occur in a controlled sequence.")
    add_figure(doc, existing_workflow, "Figure 2.1: Existing system workflow for manual email attachment analysis. Source: Author generated using draw.io, 2026.", max_width=4.6, max_height=7.2)
    add_figure(doc, EXTRACTED / "image15.png", "Figure 2.2: Proposed system workflow for StegGuard automated multi-modal detection. Source: Author generated using draw.io, 2026.", max_width=4.2, max_height=7.6)
    add_heading(doc, "2.8.8 Performance and Environment Requirements", 3)
    add_para(doc, "The development environment is a local Windows workstation running a Python-based web application and a database-backed audit layer. The system should accept PNG and WAV files, reject unsupported files such as PDF, DOC and MP3, and present a controlled error rather than exposing stack traces. Performance requirements are framed around prototype expectations: responsive single-file testing, visible batch progress and stable dashboard metrics.")
    add_heading(doc, "2.9 Algorithms Implemented", 2)
    add_para(doc, "The two core algorithms are SRM feature extraction for images and MFCC feature extraction for audio. SRM extracts residual-based statistics that can reveal pixel-level embedding changes. MFCC extraction converts the audio signal into cepstral features that summarise frequency behaviour over time. These feature sets are normalised and supplied to the detection pipeline, where the classifier produces a confidence score and verdict.")

    # Chapter 3
    doc.add_page_break()
    add_heading(doc, "3. System Analysis and Design", 1)
    add_para(doc, "System analysis and design explain how the requirements become an implementable artefact. The design uses UML diagrams, database relationships, workflow diagrams and interface wireframes. The key design decision is to treat StegGuard as a SOC system with human roles and audit controls rather than as a standalone classifier.")
    add_heading(doc, "3.1 Use Case", 2)
    add_para(doc, "The main actors are the Security Analyst, System Administrator and Compliance Auditor. The analyst uploads files, runs analysis, reviews results and exports reports. The administrator configures thresholds and manages users. The auditor retrieves logs and reviews compliance evidence. The system boundary contains the functions that belong to StegGuard, while external actors start interactions from outside the boundary.")
    add_figure(doc, usecase, "Figure 3.1: Use case diagram for the StegGuard SOC System. Source: Author generated, 2026.", max_width=6.2, max_height=4.0)
    add_heading(doc, "3.1.1 Use Case Documentation", 3)
    add_table(doc, "Table 3.1: Use case documentation. Source: Author generated, 2026.",
              ["Use case", "Actor", "Main success scenario", "Exception"],
              [
                  ["Upload and analyse single file", "Security Analyst", "Analyst selects a file, system validates, classifies and displays result.", "Unsupported file returns validation error."],
                  ["Process batch upload", "Security Analyst", "Analyst submits multiple files, valid files are queued and summary is displayed.", "No valid files returns batch error."],
                  ["Retrieve audit logs", "Compliance Auditor", "Auditor applies filters and views matching audit records.", "Unauthorised user receives access denial."],
                  ["Configure threshold", "System Administrator", "Admin enters new threshold, system validates and stores setting.", "Invalid threshold shows validation message."],
              ],
              widths=[1.6, 1.2, 2.6, 1.1])
    add_heading(doc, "3.2 Sequence Diagram", 2)
    add_para(doc, "The selected sequence diagram shows the single file analysis flow because it is the core interaction used by both the manual analysis screen and the batch worker. It includes one external actor at the top, participant lifelines, activation bars, synchronous messages, dashed return messages, an alt fragment for invalid and valid files, an opt fragment for suspicious alerts and termination marks at the bottom.")
    add_figure(doc, DOWNLOADS / "final seq1.png", "Figure 3.2: Sequence diagram for single file analysis. Source: Author generated using PlantUML, 2026.", max_width=6.3, max_height=4.8)
    add_heading(doc, "3.3 Class Diagram", 2)
    add_para(doc, "The class diagram identifies the principal structural elements of the prototype: users, uploaded files, detection pipeline, classifier model, threshold settings, detection results, reports and audit logs. Relationship labels show how classes use or produce other classes, while multiplicity explains how many records can participate. The diagram supports maintainability by separating domain objects from services and records.")
    add_figure(doc, DOWNLOADS / "final class.drawio.png", "Figure 3.3: Class diagram for the StegGuard SOC System. Source: Author generated using draw.io, 2026.", max_width=6.3, max_height=3.0)
    add_heading(doc, "3.4 Activity Diagram", 2)
    add_para(doc, "The activity diagram selected for the main report is file upload and classification. It begins with an initial node, uses rounded actions, applies guard conditions for valid and invalid paths, creates object nodes for the result and audit log entry, and ends with a final node. This diagram directly supports the test cases in Chapter 5.")
    add_figure(doc, DOWNLOADS / "activity1.drawio.png", "Figure 3.4: Activity diagram for file upload and classification. Source: Author generated using draw.io, 2026.", max_width=3.0, max_height=7.6)
    add_heading(doc, "3.5 Database Schema", 2)
    add_para(doc, "The database design stores users, roles, uploads, batches, feature extracts, models, threshold settings, detection results, alerts and audit logs. The ERD uses ten tables without attributes in order to focus on relationship structure. The detailed schema can later be expanded with keys, data types and indexes in the implementation appendix. The key requirement is that every detection result remains traceable to an upload and a user action.")
    add_figure(doc, DOWNLOADS / "final erd.drawio.png", "Figure 3.5: Entity relationship diagram for the StegGuard SOC database. Source: Author generated using draw.io, 2026.", max_width=6.3, max_height=2.2)
    add_heading(doc, "3.6 Algorithms Implemented", 2)
    add_para(doc, "The implementation separates feature extraction from classification. This makes the code easier to test because SRM, MFCC and probability-to-verdict logic can be checked independently. It also supports future model replacement because the pipeline can keep the same input and output contract while the classifier internals evolve.")
    add_heading(doc, "3.6.1 Spatial Rich Model (SRM) Feature Extraction", 3)
    add_para(doc, "The SRM process begins by accepting an image file, validating the format and converting it to a consistent colour or grayscale representation. The image dimensions are normalised, residual filters are applied, residual maps are generated and co-occurrence matrices are computed. The final feature vector is normalised and stored for classification. The algorithm is suitable for detecting subtle pixel changes that may be introduced by image steganography.")
    add_figure(doc, DOWNLOADS / "final srm flowchart.drawio.png", "Figure 3.6: SRM algorithm flowchart for image feature extraction. Source: Author generated using draw.io, 2026.", max_width=2.9, max_height=7.8)
    add_heading(doc, "3.6.2 MFCC Audio Feature Extraction", 3)
    add_para(doc, "The MFCC process validates an audio file, converts it to a mono signal, resamples the waveform, applies pre-emphasis, frames the signal, applies a window function and computes a Fourier transform. A Mel filter bank, log energy calculation and discrete cosine transform then generate the cepstral feature vector. The output is normalised before being passed to the audio classifier.")
    add_figure(doc, DOWNLOADS / "final mfcc flowchart.drawio.png", "Figure 3.7: MFCC audio feature extraction algorithm flowchart. Source: Author generated using draw.io, 2026.", max_width=2.9, max_height=7.8)
    add_heading(doc, "3.7 Detection Pipeline Flowchart", 2)
    add_para(doc, "The detection pipeline integrates the validation, modality identification, feature extraction, classification, storage and display stages. The flowchart also represents false scenarios, including unsupported files and scores below threshold. This makes the pipeline useful for both design explanation and testing because each decision point can be linked to an expected output.")
    add_figure(doc, DOWNLOADS / "final pipeline.drawio (2).png", "Figure 3.8: Detection pipeline flowchart for StegGuard SOC. Source: Author generated using draw.io, 2026.", max_width=3.6, max_height=7.8)
    add_heading(doc, "3.8 System Interface and Design", 2)
    add_para(doc, "The interface design follows SOC dashboard conventions: compact navigation, clear status labels, alert lists, processing statistics and action buttons. The design avoids a marketing-style landing page because the system is an operational tool. Users should immediately see workload, detections, model status, recent files and quick actions.")
    add_heading(doc, "3.8.1 Wireframe", 3)
    add_para(doc, "The dashboard wireframe shows metrics for files analysed, detections, model F1-score, false positive rate and processing time. It also includes the CNN-GRU detection pipeline, recent batch analysis, classifier benchmarks, precision/recall indicators, live alerts and sprint progress. The batch upload wireframe shows drag-and-drop upload, rejected files, queue status, progress, metrics and results.")
    add_figure(doc, DOWNLOADS / "dashboard_wireframe.png", "Figure 3.9: Dashboard wireframe for the StegGuard SOC System. Source: Author generated, 2026.", max_width=6.3, max_height=4.7)
    add_figure(doc, DOWNLOADS / "batch_upload_wireframe.png", "Figure 3.10: Batch upload wireframe for the StegGuard SOC System. Source: Author generated, 2026.", max_width=6.3, max_height=4.7)
    add_heading(doc, "3.8.2 Mockup (Implemented Functions)", 3)
    add_para(doc, "The implemented mockup converts the wireframe into a dark SOC-style interface with working navigation for dashboard, batch upload, analysis, model, audit logs and settings. It presents confidence values, verdict badges and audit identifiers. The visual design is intended to support repeated analyst use by prioritising scanability, status clarity and quick action over decorative elements.")

    # Chapter 4
    doc.add_page_break()
    add_heading(doc, "4. System Build and Technical Notes", 1)
    add_para(doc, "The prototype is implemented as a web application with a Python backend and a browser-based SOC interface. The backend exposes prediction, statistics and audit log endpoints. The file upload endpoint accepts multipart form data, validates the file, extracts features, obtains a score and writes the result. The statistics endpoint summarises recent scans for dashboard cards, while the audit endpoint supports review and compliance evidence.")
    add_para(doc, "The build separates validation, feature extraction, classification, database storage and audit logging into distinct responsibilities. This separation reflects the class and sequence diagrams and makes the system easier to test. It also makes future improvement practical: the classifier can be replaced, the database can be migrated and the worker process can be scaled without rewriting the entire interface.")
    add_para(doc, "A security-relevant build decision is to hash files using SHA-256 before storing metadata. The prototype does not need to retain every raw file for the dissertation demonstration. Hashing gives a stable reference for audit and duplicate detection while reducing the risk of storing suspicious content unnecessarily. Unsupported files are rejected with clear validation messages.")
    add_para(doc, "The dashboard and batch upload screens were implemented to demonstrate operational fit. The dashboard shows real-time metrics, live alerts and recent batch analysis. The batch upload screen supports queued files, progress feedback and results. The analysis screen exposes single-file feature and API response behaviour so that the tester can confirm the classifier contract.")

    # Chapter 5
    doc.add_page_break()
    add_heading(doc, "5. Testing Strategy and Evaluation", 1)
    add_para(doc, "Testing is organised around definitions, algorithm tests and user scenarios. This structure matches the updated dissertation contents and creates a clear link between the requirements, design diagrams and evidence screenshots. The purpose is not only to show that code runs, but also to show that StegGuard behaves predictably when analysts use it with valid, invalid, suspicious and borderline files.")
    add_heading(doc, "5.1 Testing Definitions and Approach", 2)
    add_table(doc, "Table 5.1: Testing definitions and approach. Source: Author generated, 2026.",
              ["Testing type", "Purpose in StegGuard", "Evidence"],
              [
                  ["Unit testing", "Check individual functions such as probability mapping and feature extraction shape.", "Pytest unit screenshot"],
                  ["Integration testing", "Check API behaviour across upload, validation, classification and statistics.", "Pytest integration screenshot"],
                  ["Scenario testing", "Check user workflows such as dashboard refresh and batch upload.", "Interface screenshots"],
                  ["Negative testing", "Confirm corrupt or unsupported files return controlled errors.", "Invalid file test cases"],
                  ["Usability inspection", "Confirm status labels, verdicts and alerts are understandable.", "Dashboard and batch UI screenshots"],
              ],
              widths=[1.4, 3.4, 1.5])
    add_heading(doc, "5.2 Algorithm Tests", 2)
    add_para(doc, "Algorithm tests verify that the implementation returns valid structures before it is trusted inside the SOC workflow. The SRM tests check that image extraction produces a dataframe, the expected number of features and no NaN values. The MFCC and probability tests check that classifier inputs and boundary verdicts behave consistently. These tests reduce the risk that the dashboard displays a result created from invalid feature data.")
    add_table(doc, "Table 5.2: Algorithm tests. Source: Author generated, 2026.",
              ["Test ID", "Area", "Expected result", "Observed result"],
              [
                  ["AT1", "Probability conversion", "High probability returns stego verdict.", "Passed"],
                  ["AT2", "Boundary conversion", "Low, review and high thresholds map correctly.", "Passed"],
                  ["AT3", "SRM feature extraction", "Image function returns expected feature structure.", "Passed"],
                  ["AT4", "Feature validity", "Extracted values contain no NaN values.", "Passed"],
                  ["AT5", "LSB embedding helper", "Output shape and dtype remain valid.", "Passed"],
              ],
              widths=[0.8, 1.7, 2.8, 1.0])
    add_figure(doc, EXTRACTED / "image21.png", "Screenshot 5.1: Unit test evidence for feature extraction and probability logic. Source: Author generated, 2026.", max_width=6.0, max_height=4.7)
    add_heading(doc, "5.3 Test Cases and Scenarios", 2)
    add_para(doc, "The scenario tests are derived from the UML diagrams. Single file analysis follows the sequence diagram; batch processing follows the batch upload activity; audit log retrieval follows the compliance workflow; dashboard refresh follows the statistics sequence. Each scenario includes a normal path and at least one false or exception path.")
    add_para(doc, "Acceptance criteria are written in analyst-facing terms. A test passes only when the system gives a useful outcome to the person using it: a valid file must return a confidence score and verdict, an invalid file must return a clear explanation, a batch must show progress and summary values, and an audit-related action must leave a retrievable record. This approach avoids testing only internal functions while ignoring whether the SOC workflow remains understandable.")
    add_table(doc, "Table 5.3: Test cases and scenarios. Source: Author generated, 2026.",
              ["Scenario", "Input or action", "Expected outcome", "Status"],
              [
                  ["Single file valid image", "Upload PNG image and click analyse.", "System returns verdict, confidence and audit ID.", "Passed"],
                  ["Single file invalid file", "Upload unsupported TXT or corrupt image.", "System returns validation error and logs event.", "Passed"],
                  ["Audio analysis", "Upload supported WAV file.", "System returns audio modality result.", "Passed"],
                  ["Batch upload", "Upload multiple PNG/WAV files.", "Valid files queue and batch summary is displayed.", "Passed"],
                  ["No valid batch files", "Upload unsupported formats only.", "System rejects batch and records audit event.", "Passed"],
                  ["Dashboard refresh", "Open dashboard after scans.", "Cards and live alerts show updated totals.", "Passed"],
                  ["Audit retrieval", "Open audit logs with filters.", "Authorised user receives matching records.", "Planned for extended evaluation"],
              ],
              widths=[1.4, 1.8, 2.4, 0.9])
    add_figure(doc, EXTRACTED / "image18.png", "Screenshot 5.2: Integration test evidence showing upload, invalid file and statistics checks. Source: Author generated, 2026.", max_width=6.0, max_height=4.7)
    add_figure(doc, EXTRACTED / "image25.png", "Screenshot 5.3: Dashboard runtime evidence showing metrics, live alerts and detection pipeline status. Source: Author generated, 2026.", max_width=6.3, max_height=3.2)
    add_figure(doc, EXTRACTED / "image27.png", "Screenshot 5.4: Batch upload runtime evidence showing upload area, format guidance and live alerts. Source: Author generated, 2026.", max_width=6.3, max_height=3.0)
    add_para(doc, "The test evidence indicates that the prototype meets the core dissertation requirements. Unit and integration tests pass for feature extraction, probability mapping, valid file uploads, invalid file handling and statistics updates. The interface screenshots show that the implemented UI reflects the wireframe design and that analysts can see verdicts, alerts, progress and operational metrics.")
    add_para(doc, "Some limitations remain. The screenshots prove that the system runs and that major workflows are visible, but a larger final evaluation should include a broader dataset, repeated timing measurements and a confusion matrix generated from labelled samples. The current interim evaluation is therefore best understood as functional and integration evidence. It is sufficient for demonstrating progress, while the final dissertation should strengthen claims about detection performance and robustness.")

    # Chapter 6
    doc.add_page_break()
    add_heading(doc, "6. Progress To Date", 1)
    add_para(doc, "The project has completed the research planning, problem definition, literature review, requirements analysis, workflow design, UML diagrams, database design and interface wireframes. Core development has also progressed: file upload validation, SRM image feature extraction, MFCC audio feature extraction, classification logic, dashboard metrics, batch upload and audit-style evidence have been implemented at prototype level.")
    add_para(doc, "The most important progress is that the dissertation is no longer limited to conceptual design. The system now has visible screens, test outputs and diagrams that align with the implemented behaviour. This improves the credibility of the interim report because the reader can trace the project from aim, to requirements, to diagrams, to implementation and tests. Remaining work is mainly refinement, evaluation depth and final dissertation writing.")
    add_para(doc, "This progress also reduces delivery risk because the remaining tasks are now focused on strengthening evidence, improving clarity and polishing the submission rather than discovering whether the proposed architecture is feasible.")

    # Chapter 7
    add_heading(doc, "7. Future Work", 1)
    add_para(doc, "Future work will focus on expanding the evaluation dataset, improving the classifier, strengthening database persistence and completing final documentation. The classifier should be tested against a wider range of clean and stego samples, including different embedding strengths and file sources. Threshold settings should be reviewed against false positive and false negative behaviour so that the review band is useful to analysts.")
    add_para(doc, "The system should also be extended with role-based access control, improved audit log filtering, report export and deployment hardening. Batch processing can be improved by moving long-running jobs to a background queue and by adding retry logic for failed files. Final dissertation work will include proofreading, reference checking, appendix organisation and preparation of presentation slides. The Gantt chart in Appendix A sets out the remaining schedule to 20 August 2026.")
    add_para(doc, "The final submission should also connect evaluation findings back to the original aim. If the system performs well on image files but less consistently on audio, that result should be discussed honestly rather than hidden. A dissertation-quality conclusion can then explain what StegGuard proves, what remains experimental and what would be required before a real organisation could adopt it.")

    # References
    doc.add_page_break()
    add_heading(doc, "References", 1)
    refs = [
        "Almhlbdi, A.M., Altowairqi, N.D., Alshutayri, A.O. and Qarout, R.K. (2025) Deep learning-based multi-class detection of LSB steganography. IEEE Access, 04 November 2025.",
        "Bas, P., Filler, T. and Pevny, T. (2011) Break Our Steganographic System. Information Hiding, Lecture Notes in Computer Science, vol. 6958.",
        "Kodovsky, J., Fridrich, J. and Holub, V. (2012) Ensemble classifiers for steganalysis of digital media. IEEE Transactions on Information Forensics and Security, 7(2), pp.432-444.",
        "Panayotov, V. et al. (2015) Librispeech: an ASR corpus based on public domain audio books. Proceedings of ICASSP.",
        "Ren, F., Wang, Y., Zhu, T. and Gao, B. (2024) Secure steganography based on Wasserstein GAN-GP. Proceedings of ICNLP, 22-24 March 2024.",
        "Samuel, H.D., Kumar, M.S., Aishwarya, R. and Mathivanan, G. (2022) Automation detection of malware and steganographical content. Proceedings of ICMMC, 29-31 March 2022.",
        "Wahono, D., De La Croix, N.J. and Ahmad, T. (2024) Enhanced steganographic scheme based on difference expansion. Proceedings of ICITISEE, 31 October 2024.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.left_indent = Inches(0.25)
        track(ref)

    # Appendices: not tracked in word count.
    doc.add_page_break()
    add_heading(doc, "Appendix A: Gantt Chart", 1)
    add_figure(doc, DOWNLOADS / "final gantt chart.png", "Figure A.1: Gantt chart for StegGuard SOC project progress and future work. Source: Author generated, 2026.", max_width=6.3, max_height=4.6, main=False)
    add_heading(doc, "Appendix B: Project Log", 1)
    add_table(doc, "Table B.1: Summary project log. Source: Author generated, 2026.",
              ["Period", "Work completed", "Evidence"],
              [
                  ["March 2026", "Research planning, topic selection and problem identification.", "Draft report Chapter 1"],
                  ["April 2026", "Literature review, requirements and workflow design.", "Chapter 2 and workflow diagrams"],
                  ["May 2026", "UML, ERD, algorithm flowcharts and wireframes.", "Chapter 3 diagrams"],
                  ["June-July 2026", "Core modules, dashboard, batch upload and testing evidence.", "Prototype screenshots and pytest evidence"],
                  ["August 2026", "Final evaluation, proofreading, formatting and submission.", "Planned final dissertation work"],
              ],
              widths=[1.2, 3.5, 1.8], main=False)
    add_heading(doc, "Appendix C: Source Code", 1)
    add_para(doc, "The main source code is organised around API endpoints, validation utilities, feature extraction functions, classifier logic, database storage and UI pages. Key endpoint examples include POST /api/predict for single-file analysis, GET /api/stats for dashboard statistics and GET /api/logs for audit record retrieval.", main=False)
    add_heading(doc, "Appendix D: Presentation Slide", 1)
    add_para(doc, "A final presentation slide deck can be prepared from the report structure, using the problem statement, proposed solution, detection pipeline, testing evidence, progress chart and future work as the core narrative.", main=False)

    return doc


if __name__ == "__main__":
    doc = build_doc()
    total = sum(word_count(t) for t in main_words)
    doc.save(OUT_DOCX)
    print(f"Saved: {OUT_DOCX}")
    print(f"Approximate main word count excluding appendices: {total}")
