RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Use ONLY the context below to answer the question.

If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""