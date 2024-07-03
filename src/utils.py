from PyPDF2 import PdfReader

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

class Config:
    def __init__(self):
        self.max_length = 2048
        self.provide_context = True

    def update(self, max_length, provide_context):
        self.max_length = max_length
        self.provide_context = provide_context