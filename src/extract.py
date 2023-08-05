import os
import PyPDF4
import pdfplumber
from langchain.text_splitter import Language

class Extract:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_metadata(self):
        raise NotImplementedError

    def extract_text(self):
        raise NotImplementedError


class PDFExtract(Extract):
    def extract_metadata(self):
        with open(self.file_path, "rb") as pdf_file:
            reader = PyPDF4.PdfFileReader(pdf_file)
            metadata = reader.getDocumentInfo()
            return {
                "title": metadata.get("/Title", "").strip(),
                "author": metadata.get("/Author", "").strip(),
                "creation_date": metadata.get("/CreationDate", "").strip(),
            }

    def extract_text(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with pdfplumber.open(self.file_path) as pdf:
            pages = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text.strip():
                    pages.append((page_num + 1, text))

        return pages


class TXTExtract(Extract):
    def __init__(self, file_path: str, language: Language):
        super().__init__(file_path)
        self.splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=1000, chunk_overlap=200
        )

    def extract_metadata(self):
        # For a text file, you might have a different way to extract metadata or have none at all
        return {}

    def extract_text(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "r") as txt_file:
            text = txt_file.read()
        chunks = self.splitter.split_text(text)
        pages = [(i+1, chunk) for i, chunk in enumerate(chunks)]  # Treat each chunk as a page

        return pages


# Add this code where the other classes (Extract, PDFExtract, TXTExtract) are defined

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

class TextSplitter():
    def split_documents(self, docs):
        raise NotImplementedError


class JSSplitter(TextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_documents(self, docs):
        return self._splitter.split_documents(docs)


class TXTSplitter(TextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.ENGLISH, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split_documents(self, docs):
        return self._splitter.split_documents(docs)
