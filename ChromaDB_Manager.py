import os
import re
import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract text and chunk using —— separator
def pdf_to_doc(pdf_path, document_name, document_type):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    full_text = re.sub(r'[\u2028\u2029\n\r]+', ' ', full_text).strip()
    sections = re.split(r'——+', full_text)
    sections = [sec.strip() for sec in sections if sec.strip()]

    documents = []
    for i, section in enumerate(sections):
        documents.append(
            Document(
                page_content=section,
                metadata={
                    "document_name": document_name,
                    "document_type": document_type,
                    "section_number": i
                }
            )
        )
    return documents

# Store documents in Chroma DB
def store_in_chroma(pdf_path, document_name, document_type, persist_directory="./chroma_phi"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    data = vectordb.get()
    existing = any(
        metadata.get("document_name") == document_name
        for metadata in data['metadatas']
    )
    if existing:
        print(f"⚠️ Document with name '{document_name}' already exists. Use update_entries() to replace it.")
        return
    docs = pdf_to_doc(pdf_path, document_name, document_type)
    vectordb.add_documents(docs)
    print(f"✅ Stored {len(docs)} chunks from '{document_name}' as '{document_type}'.")

# Read a single Document
def view_document(document_name, persist_directory="./chroma_phi"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    docs = vectordb.get(where={"document_name": document_name})
    chunks = docs['documents']
    if not chunks:
        print(f"❌ No document found with name '{document_name}'.")
        return
    print(f"📑 Contents of '{document_name}':\n")
    for i, chunk in enumerate(chunks):
        print(f"[Section {i+1}]\n{chunk}\n{'-'*50}")

# List all the documents in the DB
def list_documents(persist_directory="./chroma_phi"):
    if not os.path.exists(persist_directory):
        print("⚠️ No Chroma database found.")
        return

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    data = vectordb.get()
    seen = set()
    if not data['documents']:
        print("📭 No documents found in the vector store.")
        return

    print("📄 Documents stored:")
    for metadata in data['metadatas']:
        name = metadata.get("document_name", "Unknown")
        dtype = metadata.get("document_type", "Unknown")
        if (name, dtype) not in seen:
            print(f"• Name: {name} | Type: {dtype}")
            seen.add((name, dtype))

# Delete all entries by document name
def delete_entries_by_name(document_name, persist_directory="./chroma_phi"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    data = vectordb.get()
    matching_ids = [
        doc_id for doc_id, metadata in zip(data['ids'], data['metadatas'])
        if metadata.get("document_name") == document_name
    ]
    if matching_ids:
        vectordb.delete(ids=matching_ids)
        print(f"🧹 Deleted {len(matching_ids)} entries with document_name = '{document_name}'.")
    else:
        print(f"⚠️ No entries found with document_name = '{document_name}'.")


# Update entries by re-uploading document
def update_entries(pdf_path, document_name, document_type, persist_directory="./chroma_phi"):
    delete_entries_by_name(document_name, persist_directory)
    store_in_chroma(pdf_path, document_name, document_type, persist_directory)
    print(f"🔁 Updated entries for '{document_name}'.")

# Clear the DB
def clear_database(persist_directory="./chroma_phi"):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    data = vectordb.get()
    all_ids = data.get('ids', [])
    if not all_ids:
        print("📭 Vector DB is already empty.")
        return
    # Delete all documents
    vectordb.delete(ids=all_ids)
    print(f"🧹 Cleared {len(all_ids)} documents from the vector DB at '{persist_directory}'.")