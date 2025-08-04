from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_parse.pdf_parser import DoclingPdfParser
from docling_core.types.doc.page import TextCellUnit
from io import BytesIO
from docling.datamodel.base_models import DocumentStream

# Caminho absoluto ou relativo a partir do script atual
base_dir = Path(__file__).parent
pdf_filename = "Manual_de_Alteração_Cadastral_ICATU 1 1.pdf"
pdf_path = base_dir / pdf_filename

if not pdf_path.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")

# 🚀 Extração simples para Markdown
converter = DocumentConverter()
doc = converter.convert(str(pdf_path))
markdown = doc.document.export_to_markdown()
print(markdown)

# 🧠 Extração com coordenadas de palavras
parser = DoclingPdfParser()
pdf = parser.load(path_or_stream=str(pdf_path))

coords: list[tuple[int, str, tuple]] = []
for page_no, pred_page in pdf.iterate_pages():
    for cell in pred_page.iterate_cells(unit_type=TextCellUnit.WORD):
        coords.append((page_no, cell.text, cell.rect))

parser._load_document()

# Output opcional de coordenadas
for page_no, text, rect in coords[:10]:
    print(f"P{page_no}: '{text}' @ {rect}")
