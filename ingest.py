from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/MalayMondal_Resume.pdf")
docs = loader.load()

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Verify chunks
print(f"{len(chunks)} chunks created")

# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)

# Save vector store locally
db.save_local("vectorstore")

print("Vector store created and saved successfully!")