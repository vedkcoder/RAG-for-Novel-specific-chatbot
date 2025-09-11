import os
import PyPDF2 as pdf
import re
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt_tab')
# from textblob import TextBlob
# from huggingface_hub import login
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# login(token=os.environ.get('hf_token'))
import numpy as np
import pandas as pd
import faiss


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
    # print(len(words))
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed(model_name, batch_size, chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name = model_name)

    all_outputs = []
    for i in range(0,len(chunks), batch_size):
    
        batch = chunks[i:i+batch_size]
        outputs = embeddings.embed_query(str(batch))
        # batch_outputs = np.mean(outputs, axis=1)
        all_outputs.append(outputs)

    # print(len(all_outputs))
    return np.array(all_outputs)

def save_embeddings(embeddings):

    df = pd.DataFrame(embeddings)
    df.to_csv('embeddings.csv')
    print('Succesfully saved embeddings')

def load_embeddings():

    embeddings = np.array(pd.read_csv('embeddings.csv'))
    # print(embeddings, embeddings.shape)
    return embeddings

def indexing(dimensions, embeddings):

    index = faiss.IndexFlatL2(dimensions)
    matrix =  np.array([embedding.flatten() for embedding in embeddings]).astype('float32')
    print(matrix.shape)
    index.add(matrix)
    print(f'indexed {index.ntotal}')
    return index

def test_query(query, chunked_text):

    embeddings = HuggingFaceEmbeddings(model_name = 'google/embeddinggemma-300m')
    query_embedding = embeddings.embed_query(test)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    k = 5  # Number of closest documents to retrieve
    distances, indices = index.search(query_embedding, k)

    # Retrieve and print the most similar chunks
    print("Top similar document chunks:")
    for idx in indices[0]:
        print(chunked_text[idx]+'\n')



def run():

    filepath = "G:/coding stuff/Personal Projects/Agentic Chatbot/NovelData/"
    text = read_pdf(filepath)
    cleaned_text = clean_text(text)
    chunked_text = chunk_text(cleaned_text, 200)

    model_name = 'google/embeddinggemma-300m'
    embeddings = embed(model_name, batch_size=128, chunks=chunked_text)
    print(f'embeddings for {len(embeddings)} done')
    # print(embeddings.shape)
    # save_embeddings(embeddings)

    # embeddings = load_embeddings()
    index = indexing(768, embeddings)

    query = 'What is tha hair color of Rias?'
    test_query(query, chunked_text)

    

run()