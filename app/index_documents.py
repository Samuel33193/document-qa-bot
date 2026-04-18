import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"


def load_documents(data_dir=DATA_DIR):
    documents = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)

        if file_name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    return documents


def main():
    print("Loading documents...")
    documents = load_documents()

    print("Documents loaded:", len(documents))
    for i, d in enumerate(documents, start=1):
        print(f"\n--- Document {i} ---")
        print("Source:", d.metadata)
        print(d.page_content[:200])

    if not documents:
        raise ValueError("No documents were loaded from the data folder.")

    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)

    print("Total chunks created:", len(split_docs))

    print("\nCreating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    print(f"Saving vectorstore to '{VECTORSTORE_DIR}'...")
    vectorstore.save_local(VECTORSTORE_DIR)

    print("\nIndexing complete.")


if __name__ == "__main__":
    main()