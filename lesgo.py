import os
import shutil
import re
import fitz  # pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract and chunk into sentences, attach metadata
def extract_and_chunk_sentences(pdf_path, document_name, document_type):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    full_text = re.sub(r'[\u2028\u2029\n\r]+', ' ', full_text)
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())

    documents = []
    for sentence in sentences:
        if sentence.strip():
            documents.append(
                Document(
                    page_content=sentence.strip(),
                    metadata={
                        "document_name": document_name,
                        "document_type": document_type
                    }
                )
            )
    return documents


# Store documents with metadata into Chroma
def store_in_chroma(pdf_path, document_name, document_type, persist_directory="./chroma_phi"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    docs = extract_and_chunk_sentences(pdf_path, document_name, document_type)
    vectordb.add_documents(docs)
    print(f"✅ Stored {len(docs)} chunks from '{document_name}' as '{document_type}'.")


# Clear the entire vector database
def clear_database(persist_directory="./chroma_phi"):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"🗑️ Cleared vector database at {persist_directory}.")
    else:
        print("⚠️ Vector DB does not exist.")


# Delete all entries by document name
def delete_entries_by_name(document_name, persist_directory="./chroma_phi"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    deleted_count = vectordb.delete(filter={"document_name": document_name})
    print(f"🧹 Deleted {deleted_count} entries with document_name = '{document_name}'.")


# Replace a document's entries by reuploading it
def update_entries(pdf_path, document_name, document_type, persist_directory="./chroma_phi"):
    delete_entries_by_name(document_name, persist_directory)
    store_in_chroma(pdf_path, document_name, document_type, persist_directory)
    print(f"🔁 Updated entries for '{document_name}'.")


# List stored documents and metadata
def list_all_documents(persist_directory="./chroma_phi"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    data = vectordb.get(include=["documents", "metadatas"])

    if not data["documents"]:
        print("📂 No documents found in the vector store.")
        return

    print(f"📄 Found {len(data['documents'])} stored entries:\n")
    for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"]), start=1):
        print(f"{i}. [{meta.get('document_name')}] ({meta.get('document_type')}): {doc[:80]}...")


clear_database()
store_in_chroma("test.pdf", "perceiv", "project")
list_all_documents()
