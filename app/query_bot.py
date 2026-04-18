from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

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

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    response = llm.invoke(prompt)

    answer_text = response.content
    if isinstance(answer_text, list):
        try:
            answer_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in answer_text
            ).strip()
        except Exception:
            answer_text = str(answer_text).strip()
    else:
        answer_text = str(answer_text).strip()

    return {
        "answer": answer_text,
        "sources": retrieved_docs
    }