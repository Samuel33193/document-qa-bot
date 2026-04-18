import streamlit as st
from app.query_bot import ask_question

st.set_page_config(page_title="Document QA Bot", page_icon="📄", layout="wide")

st.title("📄 Document QA Bot")
st.write("Ask a question about your documents.")

query = st.text_input("Ask a question about your documents:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            result = ask_question(query)

        st.markdown("### 🤖 Answer")
        st.success(result["answer"])

        st.markdown("### 📄 Sources Used")
        for i, doc in enumerate(result["sources"], start=1):
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "N/A")
            chunk_preview = doc.page_content[:300].strip()

            st.info(
                f"**Source {i}:** {source} | **Page:** {page}\n\n"
                f"**Chunk Preview:**\n{chunk_preview}..."
            )
    else:
        st.warning("Please enter a question.")