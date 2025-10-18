import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# -----------------------
# 🔧 Config
# -----------------------
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "google/gemma-2b-it"
VECTOR_DIR = "docs/chroma/"
TOP_K = 3

# -----------------------
# 📄 Sample Knowledge Base
# -----------------------
docs = [
    Document(page_content="Paris is the capital of France."),
    Document(page_content="Python is a programming language used in AI."),
    Document(page_content="Gradio helps build ML apps with Python."),
    Document(page_content="Mistral is a high-performance open-source language model."),
]

# -----------------------
# 🧠 Embedding + Vector DB
# -----------------------
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if not os.path.exists(VECTOR_DIR):
    vectordb = Chroma.from_documents(docs, embedding=embedding, persist_directory=VECTOR_DIR)
    vectordb.persist()
else:
    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding)

# -----------------------
# 🧠 Load LLM
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float16, device_map="auto")

# -----------------------
# 🔁 RAG Logic
# -----------------------
def rag_pipeline(user_query):
    # Step 1: Retrieve top K relevant chunks
    results = vectordb.max_marginal_relevance_search(user_query, k=TOP_K)
    context = "\n".join([doc.page_content for doc in results])

    # Step 2: Construct prompt
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {user_query}
Answer:"""

    # Step 3: Generate with LLM
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# -----------------------
# 🎨 Gradio Interface
# -----------------------
gr_app = gr.Interface(
    fn=rag_pipeline,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question..."),
    outputs=gr.Textbox(label="Answer",lines = 5),
    title="🔍 RAG Chatbot",
    description="Ask questions and get answers based on embedded knowledge using open-source LLMs.",
)

# -----------------------
# 🚀 Launch
# -----------------------
if __name__ == "__main__":
    gr_app.launch()
