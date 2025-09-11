import os
import PyPDF2 as pdf
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
from textblob import TextBlob


def read_pdf(filepath):

    text = []
    
    for pdf_file in os.listdir(filepath):

        reader = pdf.PdfReader(filepath + pdf_file)
        # print(len(reader.pages))
        # print(type(reader.pages[10].extract_text()))

        for i in range(len(reader.pages)):
            text.append(reader.pages[i].extract_text())

    return text

def clean_text(text):

    

    text = [page.replace("\t", "") for page in text]
    text = [re.sub(r'\\s+', ' ', page) for page in text]
    text = [page.strip() for page in text]  
    text = [page for page in text if len(page) > 100]
    text = [re.sub(r'[0-9]+ \| H i g h  S c h o o l  D x D  V o l u m e  [0-9]+','', page) for page in text]
    # print(text[2])

    text = [char for page in text for char in page]
    text = ''.join(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]','', text)

    tokens = word_tokenize(text)
    tokens = ' '.join(tokens) 
    # tokens = str(TextBlob(tokens).correct())
    # print(tokens[0:1000])

    return tokens

def chunk_text(text, chunk_size):
    words = text.split()
    print(len(words))
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def run():

    filepath = "G:/coding stuff/Personal Projects/Agentic Chatbot/NovelData/"
    text = read_pdf(filepath)
    cleaned_text = clean_text(text)
    chunked_text = chunk_text(cleaned_text, 200)

run()