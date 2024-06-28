# place Tesseract-OCR folder in "C:\Program Files"
# \Tesseract-OCR\tesseract
# cv path D:\new\itsolera-assignment-1\CV.pdf

import os
import pytesseract
from pdf2image import convert_from_path
from docx import Document
import subprocess

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract"

def extract_text_from_pdf(file_path):
    text = ""
    images = convert_from_path(file_path)
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def extract_text_from_doc(file_path):
    output_file_path = file_path + ".txt"
    subprocess.run(['antiword', file_path, '-m', 'UTF-8', '-t', output_file_path])
    with open(output_file_path, 'r') as file:
        text = file.read()
    return text

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.doc':
        return extract_text_from_doc(file_path)
    else:
        raise ValueError("Unsupported file format")

# Example usage:
# file_path = "D:/new/itsolera-assignment-1/CV.pdf"

file_path = "D:/new/itsolera-assignment-1/CV.docx"
# Change to your file path
text = extract_text(file_path)
print(text)
