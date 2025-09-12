import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage

def get_query_embedding(query):

    embeddings = HuggingFaceEmbeddings(model_name = 'google/embeddinggemma-300m')
    # outputs = embeddings.embed_query(query)
    # outputs = np.array(embeddings).reshape(1, -1).astype('float32')
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
    # print(query_embedding)
    return query_embedding

def read_chunks(filename):

    chunks = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            chunks.append(line)

        file.close()

    return chunks


def load_index(filepath):

    index = faiss.read_index(filepath)
    return index

def retrieve_documents(query, index, k, chunks):

    embedding = get_query_embedding(query)
    # print(embedding)
    distances, indices = index.search(embedding, k)
    context_docs = [chunks[idx] for idx in indices[0]]
    return context_docs


def retrieve(query):

    filename = 'chunked_text.txt'
    index_filename = 'index.idx'
    chunks = read_chunks(filename)
    index = load_index(index_filename)

    context_docs = retrieve_documents(query, index, k=5, chunks=chunks)
    return context_docs


def generate_response(query, context_docs):

    llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation", temperature = 0.1, max_new_tokens = 512)
    context = "\n".join(context_docs)
    messages = [
        SystemMessage(content="You are a helpful assistant that helps people find information."),
        HumanMessage(content=f"Answer the question based on the context below. If the question can't be answered based on the context, say 'I don't know'\n\nContext: {context}\n\nQuestion: {query}")
    ]
    model = ChatHuggingFace(llm=llm)
    response = model.invoke(messages)
    print(response.content)


if __name__ == "__main__":

    query='the vanishing dragon'
    context = retrieve(query)
    print(context)
    response = generate_response(query, context_docs=context)
