import argparse
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


def set_run_font(run, latin, east_asia, size=None, bold=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_cell_border(cell, edge, value='single', size=6, color='000000'):
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.find(qn('w:tcBorders'))
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        properties.append(borders)
    tag = f'w:{edge}'
    element = borders.find(qn(tag))
    if element is None:
        element = OxmlElement(tag)
        borders.append(element)
    element.set(qn('w:val'), value)
    if value != 'nil':
        element.set(qn('w:sz'), str(size))
        element.set(qn('w:color'), color)


def format_table(table, latin, east_asia):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    properties = table._tbl.tblPr
    old_borders = properties.find(qn('w:tblBorders'))
    if old_borders is not None:
        properties.remove(old_borders)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'bottom', 'left', 'right', 'insideH', 'insideV'):
        element = OxmlElement(f'w:{edge}')
        if edge in ('top', 'bottom'):
            element.set(qn('w:val'), 'single')
            element.set(qn('w:sz'), '12')
            element.set(qn('w:color'), '000000')
        else:
            element.set(qn('w:val'), 'nil')
        borders.append(element)
    properties.append(borders)

    columns = len(table.columns)
    font_size = 6.5 if columns >= 10 else 7.2 if columns >= 8 else 8.0 if columns >= 7 else 8.5
    for row_index, row in enumerate(table.rows):
        row_properties = row._tr.get_or_add_trPr()
        cant_split = OxmlElement('w:cantSplit')
        row_properties.append(cant_split)
        if row_index == 0:
            repeat = OxmlElement('w:tblHeader')
            repeat.set(qn('w:val'), 'true')
            row_properties.append(repeat)
        for column_index, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            margin = cell._tc.get_or_add_tcPr().find(qn('w:tcMar'))
            if margin is None:
                margin = OxmlElement('w:tcMar')
                cell._tc.get_or_add_tcPr().append(margin)
            for side in ('top', 'left', 'bottom', 'right'):
                node = margin.find(qn(f'w:{side}'))
                if node is None:
                    node = OxmlElement(f'w:{side}')
                    margin.append(node)
                node.set(qn('w:w'), '35' if side in ('left', 'right') else '20')
                node.set(qn('w:type'), 'dxa')
            if row_index == 0:
                set_cell_border(cell, 'bottom', size=8)
            for paragraph in cell.paragraphs:
                paragraph.alignment = (WD_ALIGN_PARAGRAPH.CENTER if row_index == 0 or column_index >= 2
                                       else WD_ALIGN_PARAGRAPH.LEFT)
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.paragraph_format.keep_with_next = row_index < len(table.rows) - 1
                for run in paragraph.runs:
                    set_run_font(run, latin, east_asia, font_size,
                                 bold=True if row_index == 0 else None)


def add_page_number(section, latin, east_asia):
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement('w:fldChar')
    begin.set(qn('w:fldCharType'), 'begin')
    instruction = OxmlElement('w:instrText')
    instruction.set(qn('xml:space'), 'preserve')
    instruction.text = ' PAGE '
    end = OxmlElement('w:fldChar')
    end.set(qn('w:fldCharType'), 'end')
    run._r.extend((begin, instruction, end))
    set_run_font(run, latin, east_asia, 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--language', choices=('en', 'zh'), required=True)
    args = parser.parse_args()
    latin = 'Times New Roman'
    east_asia = 'SimSun' if args.language == 'zh' else latin
    document = Document(args.input)
    for section in document.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(1.8)
        section.bottom_margin = Cm(1.8)
        section.left_margin = Cm(1.8)
        section.right_margin = Cm(1.8)
        add_page_number(section, latin, east_asia)

    styles = {style.name: style for style in document.styles}
    for name in ('Normal', 'Body Text', 'First Paragraph', 'Compact', 'Bibliography'):
        style = styles[name]
        style.font.name = latin
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), east_asia)
        style.font.size = Pt(10.5 if name != 'Bibliography' else 8.5)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.line_spacing = 1.12
        style.paragraph_format.space_after = Pt(5 if name != 'Bibliography' else 2)
    for name, size in (('Title', 16), ('Heading 1', 13), ('Heading 2', 11.5), ('Heading 3', 10.5)):
        style = styles[name]
        style.font.name = latin
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), east_asia)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(10 if name != 'Title' else 0)
        style.paragraph_format.space_after = Pt(5)
    styles['Title'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    figure_prefixes = ('Figure ',) if args.language == 'en' else ('图 ',)
    table_prefixes = ('Table ',) if args.language == 'en' else ('表 ',)
    references = False
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text in ('References', '参考文献'):
            references = True
        if paragraph._p.xpath('.//w:drawing'):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(5)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.keep_with_next = True
        if text.startswith(figure_prefixes):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.space_after = Pt(7)
            paragraph.paragraph_format.keep_with_next = False
            for run in paragraph.runs:
                set_run_font(run, latin, east_asia, 8.5)
        elif text.startswith(table_prefixes):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_before = Pt(7)
            paragraph.paragraph_format.space_after = Pt(3)
            paragraph.paragraph_format.keep_with_next = True
            for run in paragraph.runs:
                set_run_font(run, latin, east_asia, 8.5)
        elif references and text and text not in ('References', '参考文献'):
            paragraph.paragraph_format.left_indent = Cm(0.55)
            paragraph.paragraph_format.first_line_indent = Cm(-0.55)
            paragraph.paragraph_format.space_after = Pt(2)
            for run in paragraph.runs:
                set_run_font(run, latin, east_asia, 8.5)

    for index, shape in enumerate(document.inline_shapes):
        maximum = Cm(14.0) if index == 0 else Inches(6.55)
        if shape.width > maximum:
            ratio = maximum / shape.width
            shape.width = maximum
            shape.height = int(shape.height * ratio)
    for table in document.tables:
        format_table(table, latin, east_asia)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    print(f'Saved: {output}; tables={len(document.tables)}; figures={len(document.inline_shapes)}')


if __name__ == '__main__':
    main()
