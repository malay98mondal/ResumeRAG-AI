from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import streamlit as st

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

llm = OllamaLLM(model="llama3.2")
st.title("Resume Chatbot")

question = st.text_input("Ask about resume")

if question:
    docs = db.similarity_search(question)

    context = "\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
    Resume:
    {context}

    Question:
    {question}
    """

    answer = llm.invoke(prompt)

    st.write(answer)