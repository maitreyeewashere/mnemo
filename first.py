import ollama
import numpy as np

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

dataset = []

with open('catfax.txt',encoding='utf-8') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

VECTOR_DB = []

def addChunk(chunk):
    embedding = ollama.embed(model = EMBEDDING_MODEL, input = chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    addChunk(chunk)
    print(f'added chunk {i+1}/{len(dataset)} to db')

def cosineSim(a, b):
    a = np.array(a)  
    b = np.array(b)  
    dot = np.dot(a, b)
    a_norm = np.linalg.norm(a)  
    b_norm = np.linalg.norm(b)
    return dot / (a_norm * b_norm)




def retrieve(query, top_n=3):
    query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0] 
    sims = []

    for chunk, emb in VECTOR_DB:
        sim = cosineSim(query_emb, emb) 
        sims.append((chunk, sim))

    sims.sort(key=lambda x: x[1], reverse=True)  # higher similarity at the top
    return sims[:top_n]

input_query = input("Ask the bot: ")
retrieved = retrieve(input_query)

print("Here's what i got: ")

for chunk, sim in retrieved:
    print(f' - (similarity: {sim:.2f}) {chunk}')

context = '\n'.join([f' - {chunk}' for chunk, _ in retrieved])
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context}'''


stream = ollama.chat(
    model = LANGUAGE_MODEL,
    messages = [
        {'role':'system','content':instruction_prompt},
        {'role':'user','content':input_query},
    ],
    stream = True,
)

print("Chatbot response: ")
for chunk in stream:
    print(chunk['message']['content'], end='',flush = True)