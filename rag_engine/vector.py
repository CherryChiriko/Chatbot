import os
import json
import hashlib
import pandas as pd
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------- CONFIG ----------
CHROMA_DIR = "./chroma_langchain_db"
COLLECTION_NAME = "reference_docs"
EMBED_MODEL = "mxbai-embed-large"

# ---------- EMBEDDINGS ----------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# ---------- VECTOR STORE ----------
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
# ---------- ID GENERATION ----------
def generate_id(doc):
    file_source = doc.metadata.get("source", "unknown")
    content_hash = hashlib.md5(
    (doc.page_content + str(doc.metadata)).encode()
).hexdigest()
    return f"{file_source}_{content_hash}"

# ---------- FILE PARSER ----------
def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    documents = []

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i}
                    )
                )

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            text = " | ".join([f"{col}={row[col]}" for col in df.columns])
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "row": idx}
                )
            )

    elif ext in [".xlsx", ".xls"]:
        sheets = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in sheets.items():
            for idx, row in df.iterrows():
                text = " | ".join([f"{col}={row[col]}" for col in df.columns])
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "sheet": sheet_name,
                            "row": idx
                        }
                    )
                )

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle list of records
        if isinstance(data, list):
            for i, item in enumerate(data):
                text = json.dumps(item, indent=2)
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "record": i}
                    )
                )
        else:
            text = json.dumps(data, indent=2)
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path}
                )
            )

    else:
        print(f"Unsupported file type: {ext}")

    return documents


# ---------- SPLITTER ----------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


MANIFEST_FILE = "vector_manifest.json"

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    docs_folder = "./docs"
    manifest = load_manifest()
    
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)
        if not os.path.isfile(file_path): continue

        # 1. Quick Hash Check
        raw_docs = load_file(file_path)
        if not raw_docs: continue
        
        file_content = "".join([d.page_content for d in raw_docs])
        new_hash = hashlib.md5(file_content.encode()).hexdigest()

        # Get file history from manifest
        file_entry = manifest.get(file_path, {"v1": None, "v2": None, "latest": 0})
        
        # Check if this exact content already exists in either version
        if new_hash == file_entry["v1"] or new_hash == file_entry["v2"]:
            print(f"⏩ Skipping {filename} - Content already exists in history.")
            continue

        print(f"--- Updating {filename} ---")
        
        # 2. Version Rotation Logic
        if file_entry["latest"] == 0:
            target_version = 1
        elif file_entry["latest"] == 1:
            target_version = 2
        else:
            # We already have 2 versions. Discard V1, move V2 to V1, New is V2.
            print(f"♻️ Rotating versions for {filename}. Deleting oldest...")
            
            # Delete V1 from Chroma
            vector_store.delete(where={"source": file_path, "version": 1})
            
            # Update V1 to V2's old content in manifest and Chroma
            # Note: In a real DB, you'd 'update' metadata, but here we'll 
            # simply treat the "slots" logically.
            file_entry["v1"] = file_entry["v2"]
            target_version = 2

        # 3. Add to Vector Store
        chunks = split_documents(raw_docs)
        for c in chunks:
            c.metadata["version"] = target_version
        
        # Unique IDs including version to prevent collisions
        ids = [f"{generate_id(c)}_v{target_version}" for c in chunks]
        vector_store.add_documents(chunks, ids=ids)

        # 4. Update Manifest
        file_entry[f"v{target_version}"] = new_hash
        file_entry["latest"] = target_version
        manifest[file_path] = file_entry
        save_manifest(manifest)
        
        print(f"✅ {filename} added as Version {target_version}")

    print("\n--- Sync Complete ---")