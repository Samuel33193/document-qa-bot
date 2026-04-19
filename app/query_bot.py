from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.prompt_template import RAG_PROMPT_TEMPLATE


def format_context(docs):
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()

        context_parts.append(
            f"Source: {source}, Page: {page}\nContent: {content}"
        )
    return "\n\n".join(context_parts)


def ask_question(question: str, k: int = 3):
    load_dotenv(override=True)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    retrieved_docs = vectorstore.similarity_search(question, k=k)

    print("Retrieved docs:", len(retrieved_docs))
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Retrieved Doc {i} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:300])

    if not retrieved_docs:
        return {
            "answer": "I couldn't find relevant information in the provided documents.",
            "sources": []
        }

    context = format_context(retrieved_docs)

    answer_text = (
        f"Question: {question}\n\n"
        f"Based on the retrieved document chunks, here is the relevant information:\n\n"
        f"{context}"
    )

    return {
        "answer": answer_text,
        "sources": retrieved_docs
    }