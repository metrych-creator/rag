from PyPDF2 import PdfReader
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text):
    """Remove multiple spaces and newlines."""
    if not text:
        return ""
    
    # Replace newlines with spaces
    text = text.replace("\n", " ") 

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove hyphenation at line breaks
    text = re.sub(r'(\w+)\s*-\s*(\w+)', r'\1\2', text)
    return text.strip()


def get_pdf_as_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    reader = PdfReader(pdf_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = clean_text(page.extract_text())
        if not text:
            continue

        page_doc = Document(
            page_content=text,
            metadata={
                "page_number": i + 1,
                "mediabox": str(page.mediabox),
                "rotation": page.get("/Rotate"),
                "section_title": page.get("/StructParents"),
            }
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        page_chunks = splitter.split_documents([page_doc])
        documents.extend(page_chunks)
    
    return documents

