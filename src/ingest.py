import re
import os
import sys
from typing import Callable, List, Tuple, Dict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import Language

from dotenv import load_dotenv
from extract import Extract, PDFExtract, TXTExtract


def parse_document(file_path: str, document_type) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    :param file_path: The path to the file.
    :param document_type: The type of document to parse. Default to PDF.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    parser = document_type(file_path) if callable(document_type) else document_type(file_path)
    metadata = parser.extract_metadata()
    pages = parser.extract_text()

    return pages, metadata


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []

    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


if __name__ == "__main__":
    load_dotenv()

    # Step 1: Parse document
    file_path = "src/data/chat.tsx"
    document_type = PDFExtract if file_path.endswith('.pdf') else lambda file_path: TXTExtract(file_path, Language.JS)

    raw_pages, metadata = parse_document(file_path, document_type)

    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text, metadata)

    # Optional: Reduce embedding cost by only using the first 23 pages
    # document_chunks = document_chunks[:70]

    # Step 3 + 4: Generate embeddings and store them in DB
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name="april-2023-economic",
        persist_directory="src/data/chroma",
    )

    # Save DB locally
    vector_store.persist()
