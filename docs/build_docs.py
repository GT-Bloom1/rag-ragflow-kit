"""由 docs/RAG_完整教程.md 生成 PDF 和 Word 两个版本（正文 + 附录：示例资料与测试问题，不含源代码）。

依赖：pandoc；PDF 另需 Chrome / Chromium。在项目根目录运行：
    python docs/build_docs.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BUILD = DOCS / "_build"
OUT_PDF = DOCS / "RAG_完整教程.pdf"
OUT_DOCX = DOCS / "RAG_完整教程.docx"

MATERIALS = {
    "01_机器学习基础.md": "机器学习定义，监督 / 无监督学习，过拟合，模型评估指标",
    "02_神经网络与训练.md": "感知机，激活函数，反向传播，学习率与优化器，正则化，CNN",
    "03_大语言模型与幻觉.md": "Transformer，预训练与微调，模型幻觉的成因与缓解方法",
    "04_检索增强生成RAG.md": "RAG 流程，分块，Embedding，全文与混合检索，重排序，评估",
    "05_课程安排与常见问题.md": "课程编号、教室、考核方式、作业提交等 FAQ（含编号类术语）",
}


def appendix() -> str:
    lines = ["", "# 附录 示例资料与测试问题 {.appendix}", "",
             "## 示例课程资料（`sample_docs/`）", "",
             "| 文件 | 内容 | 字数 |", "|---|---|---|"]
    for name, desc in MATERIALS.items():
        n = len((ROOT / "sample_docs" / name).read_text(encoding="utf-8"))
        lines.append(f"| {name} | {desc} | {n} |")
    lines += ["", "## 测试问题（`eval/questions.jsonl`）", "",
              "“类型”一列中，语义类问题换了原文的说法，术语类问题包含术语、数字或编号。"
              "标 ★ 的问题答案跨越多句，评测时要求这些句子出现在同一个片段中。", "",
              "| 编号 | 类型 | 问题 | 参考答案 |", "|---|---|---|---|"]
    for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            kind = "语义类" if q["type"] == "semantic" else "术语类"
            star = " ★" if q.get("evidence_all") else ""
            lines.append(f"| {q['id']}{star} | {kind} | {q['question']} | {q['answer']} |")
    return "\n".join(lines) + "\n"


def full_markdown() -> str:
    return (DOCS / "RAG_完整教程.md").read_text(encoding="utf-8").rstrip() + "\n" + appendix()


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #888; } }
body { font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; font-size: 10.5pt; line-height: 1.65;
       color: #222; max-width: none; margin: 0; }
header#title-block-header { text-align: center; margin: 40mm 0 20mm; page-break-after: always; }
header .title { font-size: 26pt; color: #c0392b; margin-bottom: 6mm; }
header .subtitle { font-size: 14pt; color: #555; }
header .author, header .date { font-size: 11pt; color: #666; margin: 2mm 0; }
nav#TOC { page-break-after: always; }
nav#TOC ul { list-style: none; padding-left: 1.2em; }
nav#TOC > ul { padding-left: 0; }
nav#TOC a { color: #222; text-decoration: none; }
h1 { color: #c0392b; font-size: 18pt; border-bottom: 2px solid #c0392b; padding-bottom: 2mm; margin-top: 10mm; }
h1.appendix { page-break-before: always; }
h2 { color: #2471a3; font-size: 14pt; margin-top: 7mm; }
h3 { color: #5d8a1c; font-size: 11.5pt; margin-top: 5mm; }
h1, h2, h3 { page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 9.5pt; }
tr, img { page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 4px 7px; vertical-align: top; }
th { background: #f2f4f7; }
code { font-family: "Noto Sans Mono CJK SC", "Consolas", monospace; font-size: 9pt; background: #f5f5f5;
       padding: 0 2px; border-radius: 2px; }
pre { background: #f7f7f9; border: 1px solid #e1e1e8; border-radius: 4px; padding: 8px 10px; font-size: 8.3pt;
      line-height: 1.45; white-space: pre-wrap; word-break: break-all; overflow-wrap: anywhere; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { border-left: 4px solid #f0b429; background: #fffbea; margin: 3mm 0; padding: 2mm 4mm; color: #444; }
img { max-width: 100%; }
figure { margin: 4mm 0; text-align: center; }
figcaption { font-size: 9pt; color: #666; }
a { color: #2471a3; }
"""


def find_chrome() -> str:
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if shutil.which(name):
            return shutil.which(name)
    sys.exit("找不到 Chrome / Chromium，无法生成 PDF")


def build_pdf(md: Path) -> None:
    html, css = BUILD / "full.html", BUILD / "style.css"
    css.write_text(CSS, encoding="utf-8")
    subprocess.run(["pandoc", str(md), "-f", "markdown", "-t", "html5", "-s", "--columns=10000", "--toc", "--toc-depth=2",
                    "--highlight-style=tango", "--embed-resources", f"--css={css}",
                    f"--resource-path={DOCS}", "-M", "lang=zh-CN", "-M", "toc-title=目录",
                    "-o", str(html)], check=True, stderr=subprocess.DEVNULL)
    subprocess.run([find_chrome(), "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"已生成 {OUT_PDF}")


# ---------------- Word ----------------
FONT_CN, FONT_EN, FONT_CODE = "Microsoft YaHei", "Calibri", "Consolas"
HEADING_STYLE = {"Heading1": ("C0392B", 32), "Heading2": ("2471A3", 28), "Heading3": ("5D8A1C", 24)}
EXTRA_STYLES = f"""
  <w:style w:type="paragraph" w:customStyle="1" w:styleId="SourceCode">
    <w:name w:val="Source Code"/><w:basedOn w:val="Normal"/><w:link w:val="VerbatimChar"/>
    <w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F5F6F8"/>
      <w:spacing w:before="0" w:after="0" w:line="260" w:lineRule="auto"/><w:ind w:left="120" w:right="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="{FONT_CODE}" w:hAnsi="{FONT_CODE}" w:eastAsia="{FONT_CN}"/><w:sz w:val="18"/></w:rPr>
  </w:style>
"""
TABLE_BORDERS = ('<w:tblBorders>' + "".join(
    f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
    for side in ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>')


def rfonts(cn: str = FONT_CN, en: str = FONT_EN) -> str:
    return f'<w:rFonts w:ascii="{en}" w:hAnsi="{en}" w:eastAsia="{cn}" w:cs="{en}"/>'


def edit_style(xml: str, sid: str, fn) -> str:
    """只在 styleId 对应的那一个 <w:style> 块内做修改。"""
    pat = re.compile(rf'<w:style [^>]*w:styleId="{sid}"[^>]*>.*?</w:style>', re.S)
    return pat.sub(lambda m: fn(m.group(0)), xml, count=1)


def patch_styles(xml: str) -> str:
    xml = re.sub(r'<w:rPrDefault>.*?</w:rPrDefault>',
                 f'<w:rPrDefault><w:rPr>{rfonts()}<w:sz w:val="21"/><w:szCs w:val="21"/>'
                 '<w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr></w:rPrDefault>', xml, flags=re.S)
    for sid, (color, size) in HEADING_STYLE.items():
        def fix(block, color=color, size=size):
            block = re.sub(r'<w:rFonts [^>]*/>', rfonts(), block)
            block = re.sub(r'<w:color [^>]*/>', f'<w:color w:val="{color}"/>', block)
            block = re.sub(r'<w:sz w:val="\d+" ?/>', f'<w:sz w:val="{size}"/>', block)
            return re.sub(r'<w:szCs w:val="\d+" ?/>', f'<w:szCs w:val="{size}"/>', block)
        xml = edit_style(xml, sid, fix)
    for sid in ("Title", "Subtitle", "TOCHeading"):
        xml = edit_style(xml, sid, lambda b: re.sub(r'<w:rFonts [^>]*/>', rfonts(), b))
    xml = edit_style(xml, "Title", lambda b: re.sub(r'<w:color [^>]*/>', '<w:color w:val="C0392B"/>', b))
    xml = edit_style(xml, "VerbatimChar", lambda b: re.sub(
        r'<w:rPr>.*?</w:rPr>', f'<w:rPr><w:rFonts w:ascii="{FONT_CODE}" w:hAnsi="{FONT_CODE}" w:eastAsia="{FONT_CN}"/>'
        '<w:sz w:val="19"/><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/></w:rPr>', b, flags=re.S))

    def table(b: str) -> str:
        b = b.replace("<w:tblCellMar>", TABLE_BORDERS + "<w:tblCellMar>", 1)
        b = re.sub(r'(<w:tblStylePr w:type="firstRow">)', r'\1<w:rPr><w:b/></w:rPr>', b)
        return re.sub(r'<w:tcPr>.*?</w:tcPr>', '<w:tcPr><w:tcBorders><w:bottom w:val="single"/></w:tcBorders>'
                      '<w:shd w:val="clear" w:color="auto" w:fill="EEF2F7"/><w:vAlign w:val="bottom"/></w:tcPr>',
                      b, flags=re.S)
    xml = edit_style(xml, "Table", table)
    xml = edit_style(xml, "BlockText", lambda b: b.replace(
        "<w:pPr>", '<w:pPr><w:pBdr><w:left w:val="single" w:sz="18" w:space="6" w:color="F0B429"/></w:pBdr>'
        '<w:shd w:val="clear" w:color="auto" w:fill="FFF8E1"/>', 1))
    return xml.replace("</w:styles>", EXTRA_STYLES + "</w:styles>")


FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r>'
          '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
SECT_PR = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdClawsFooter"/>'
           '<w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1304" w:right="1134" w:bottom="1304" w:left="1134" w:header="709" w:footer="709" w:gutter="0"/>'
           '</w:sectPr>')


def rewrite_zip(src: Path, dst: Path, edit) -> None:
    """复制 docx，edit(name, data) 返回新内容（None 表示不改）；可通过 edit.extra 追加新文件。"""
    tmp = dst.with_suffix(".tmp")
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            new = edit(item.filename, data.decode("utf-8") if item.filename.endswith(".xml") or
                       item.filename.endswith(".rels") else data)
            zout.writestr(item, new if new is not None else data)
        for name, content in getattr(edit, "extra", {}).items():
            zout.writestr(name, content)
    tmp.replace(dst)


def build_reference() -> Path:
    ref = BUILD / "reference.docx"
    subprocess.run(["pandoc", "-o", str(ref), "--print-default-data-file", "reference.docx"], check=True)
    rewrite_zip(ref, ref, lambda name, data: patch_styles(data) if name == "word/styles.xml" else None)
    return ref


TEXT_WIDTH_TWIPS = 11906 - 2 * 1134          # A4 宽度减去左右页边距
TEXT_WIDTH_EMU = TEXT_WIDTH_TWIPS * 635


def _display_len(text: str) -> int:
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def fit_tables_and_images(xml: str) -> str:
    """pandoc 把表格列设成等宽、图片按 Letter 版心缩放；这里按内容长度重分列宽，并让图片占满 A4 版心。"""
    from lxml import etree
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    w = "{%s}" % ns["w"]
    root = etree.fromstring(xml.encode("utf-8"))
    for tbl in root.iter(w + "tbl"):
        rows = tbl.findall("w:tr", ns)
        ncol = max(len(r.findall("w:tc", ns)) for r in rows)
        need = [4] * ncol
        for r in rows:
            for i, tc in enumerate(r.findall("w:tc", ns)):
                text = "".join(tc.itertext())
                need[i] = max(need[i], min(_display_len(text), 60))
        widths = [max(int(TEXT_WIDTH_TWIPS * n / sum(need)), 900) for n in need]
        widths[-1] += TEXT_WIDTH_TWIPS - sum(widths)
        for col, width in zip(tbl.find("w:tblGrid", ns).findall("w:gridCol", ns), widths):
            col.set(w + "w", str(width))
        tblw = tbl.find("w:tblPr/w:tblW", ns)
        tblw.set(w + "type", "dxa")
        tblw.set(w + "w", str(TEXT_WIDTH_TWIPS))
        for r in rows:
            for i, tc in enumerate(r.findall("w:tc", ns)):
                tcpr = tc.find("w:tcPr", ns)
                if tcpr is None:
                    tcpr = etree.SubElement(tc, w + "tcPr")
                    tc.insert(0, tcpr)
                tcw = tcpr.find("w:tcW", ns)
                if tcw is None:
                    tcw = etree.Element(w + "tcW")
                    tcpr.insert(0, tcw)
                tcw.set(w + "type", "dxa")
                tcw.set(w + "w", str(widths[i]))
    xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True).decode("utf-8")

    def scale(m):
        cx, cy = int(m.group(2)), int(m.group(3))
        return f'{m.group(1)}cx="{TEXT_WIDTH_EMU}" cy="{cy * TEXT_WIDTH_EMU // cx}"'
    return re.sub(r'(<(?:wp:extent|a:ext) )cx="(\d+)" cy="(\d+)"', scale, xml)


def postprocess_docx(path: Path) -> None:
    def edit(name, data):
        if name == "word/document.xml":
            body = re.sub(r'<w:sectPr\b.*?</w:sectPr>|<w:sectPr\s*/>', "", fit_tables_and_images(data), flags=re.S)
            return body.replace("</w:body>", SECT_PR + "</w:body>")
        if name == "word/_rels/document.xml.rels":
            return data.replace("</Relationships>", '<Relationship Id="rIdClawsFooter" Type="http://schemas.'
                                'openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
                                "</Relationships>")
        if name == "[Content_Types].xml":
            if 'Extension="png"' not in data:
                data = data.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
            return data.replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/'
                                'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
        return None
    edit.extra = {"word/footer1.xml": FOOTER}
    rewrite_zip(path, path, edit)


def build_docx(md: Path) -> None:
    subprocess.run(["pandoc", str(md), "-f", "markdown", "-t", "docx", "--columns=10000", "--toc", "--toc-depth=2",
                    f"--reference-doc={build_reference()}", "--highlight-style=tango", f"--resource-path={DOCS}",
                    "-M", "lang=zh-CN", "-M", "toc-title=目录", "-o", str(OUT_DOCX)], check=True,
                   stderr=subprocess.DEVNULL)
    postprocess_docx(OUT_DOCX)
    print(f"已生成 {OUT_DOCX}")


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    md = BUILD / "full.md"
    md.write_text(full_markdown(), encoding="utf-8")
    build_docx(md)
    if "--docx-only" not in sys.argv:
        build_pdf(md)


if __name__ == "__main__":
    main()
