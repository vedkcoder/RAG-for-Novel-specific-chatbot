import os
import PyPDF2 as pdf
import re
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt_tab')
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np
import pandas as pd
import faiss


def read_pdf(filepath):

    text = []
    
    for pdf_file in os.listdir(filepath):

        reader = pdf.PdfReader(filepath + pdf_file)

        for i in range(len(reader.pages)):
            text.append(reader.pages[i].extract_text())

    return text

def clean_text(text):

    

    text = [page.replace("\t", "") for page in text]
    text = [re.sub(r'\\s+', ' ', page) for page in text]
    text = [page.strip() for page in text]  
    text = [page for page in text if len(page) > 100]

    text = [char for page in text for char in page]
    text = ''.join(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]','', text)

    tokens = word_tokenize(text)
    tokens = ' '.join(tokens) 

    return tokens

def chunk_text(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed(model_name, batch_size, chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name = model_name)

    all_outputs = []
    for i in range(0,len(chunks), batch_size):
    
        batch = chunks[i:i+batch_size]
        outputs = embeddings.embed_query(str(batch))
        all_outputs.append(outputs)

    return np.array(all_outputs)

def save_embeddings(embeddings, embeddings_filepath):

    df = pd.DataFrame(embeddings)
    df.to_csv(embeddings_filepath, index=False)
    print('Succesfully saved embeddings')

def load_embeddings():

    embeddings = pd.read_csv('embeddings.csv')
    embeddings = np.array(embeddings.iloc[:, 1:].values)
    return embeddings

def indexing(dimensions, embeddings):

    index = faiss.IndexFlatL2(dimensions)
    matrix =  np.array([embedding.flatten() for embedding in embeddings]).astype('float32')
    print(matrix.shape)
    index.add(matrix)
    return index

def save_index(index, index_filepath):
    
    faiss.write_index(index, index_filepath)
    print('wrote indices to file')

def save_chunks(chunked_text, chunk_filename):

    chunked_text = "\n".join(chunked_text)
    with open(chunk_filename, 'w') as file:
        file.write(chunked_text)
        file.close()

    

def test_query(query, chunked_text, index):

    embeddings = HuggingFaceEmbeddings(model_name = 'google/embeddinggemma-300m')
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    k = 5
    distances, indices = index.search(query_embedding, k)

    print("Top similar document chunks:")
    for idx in indices[0]:
        print(chunked_text[idx]+'\n')



def generate_vectordb():

    filepath = "G:/coding stuff/Personal Projects/Agentic Chatbot/NovelData/"
    text = read_pdf(filepath)
    cleaned_text = clean_text(text)
    chunked_text = chunk_text(cleaned_text, 200)

    if not os.path.exists('chunked_text.txt'):
        chunk_filename = 'chunked_text.txt'
        save_chunks(chunked_text, chunk_filename)

    model_name = 'google/embeddinggemma-300m'
    embeddings = embed(model_name, batch_size=32, chunks=chunked_text)
    print(f'embeddings for {len(embeddings)} done')

    embeddings_filepath = 'embeddings.csv'
    if not os.path.exists(embeddings_filepath):
        save_embeddings(embeddings, embeddings_filepath)
    else:
        embeddings = load_embeddings()

    index = indexing(embeddings.shape[1], embeddings)

    index_filepath = 'faiss_index.idx'
    if not os.path.exists(index_filepath): 
        save_index(index, index_filepath)


generate_vectordb()