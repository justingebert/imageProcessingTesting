# recognition.py

import easyocr
import pytesseract

def recognize_with_easyocr(image_path):
    reader = easyocr.Reader(lang_list=['en'])
    return reader.readtext(image_path)

def recognize_with_pytesseract(image_path):
    return pytesseract.image_to_string(image_path)
