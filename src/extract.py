import os
import PyPDF4
import pdfplumber

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
    def extract_metadata(self):
        # For a text file, you might have a different way to extract metadata or have none at all
        return {}

    def extract_text(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "r") as txt_file:
            text = txt_file.read()
        pages = [(1, text)]  # For simple text files, we can treat whole content as one page

        return pages
