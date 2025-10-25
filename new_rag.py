# rag_local.py

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------
# Step 1: Fetch URL content
# ---------------------------
def fetch_url_content(url):
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'} 
    response = requests.get(url,headers=headers)
#    text=BeautifulSoup(webpage,'lxml')
#    return text
    soup = BeautifulSoup(response.text, "html.parser")
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text
   
# ---------------------------
# Step 2: Split text into chunks
# ---------------------------
def split_text(text, chunk_size=500, overlap=50):
    # print(text)
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # local model

def get_embeddings(texts):
    return embed_model.encode(texts, show_progress_bar=True)

# ---------------------------
# Step 4: Build FAISS index
# ---------------------------
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ---------------------------
# Step 5: Retrieve relevant chunks
# ---------------------------
def retrieve_chunks(query, index, chunks, top_k=1, threshold=0.6):
    query_emb = get_embeddings([query])
    distances, indices = index.search(np.array(query_emb).astype('float32'), top_k)
    return [chunks[i] for i in indices[0]]

def extract_relevant_sentences(chunk, query, top_k=3):
    sentences = chunk.split('. ')
    sentence_embeddings = get_embeddings(sentences)
    query_emb = get_embeddings([query])
    
    # Compute cosine similarity
    scores = np.dot(np.array(sentence_embeddings), np.array(query_emb).T).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    
    return '. '.join([sentences[i] for i in top_indices])


# ---------------------------
# Step 6: Load local LLM
# ---------------------------
model_name = "distilgpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def answer_question(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = text.replace(prompt, "").strip()
    return answer


if __name__ == "__main__":
    url ="https://sohanmaali.vercel.app/"
    text = fetch_url_content(url)
    chunks = split_text(text)
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(np.array(embeddings).astype('float32'))

    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        retrieved = retrieve_chunks(query, index, chunks)
        if retrieved:
            relevant_text = extract_relevant_sentences(retrieved[0], query)
          
            answer = answer_question(query, [relevant_text])
        else:
            answer = "Sorry, I don't know the answer."
                    
        print("\nAnswer ============ :\n", answer)
 