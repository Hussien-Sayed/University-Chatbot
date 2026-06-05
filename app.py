import os
import streamlit as st
from dotenv import load_dotenv

from src.data_utils.data_loader import DataLoader
from src.llm.embedding_api import EmbeddingAPI
from src.llm.llm_api import LLMAPI
from src.rag.vector_db.vector_db_builder import VectorDBBuilder
from src.rag.retriever.rag_retriever import RAGRetriever


load_dotenv()


@st.cache_resource
def build_vector_db():
    data_loader = DataLoader()
    embedding_api = EmbeddingAPI()

    builder = VectorDBBuilder(
        data_loader=data_loader,
        embedding_api=embedding_api,
        vdb_save_path=os.getenv("VDB_SAVE_PATH", "data/vector_db"),

    )

    result = builder.build_vector_db()
    return result


@st.cache_resource
def load_retriever():
    llm_api = LLMAPI()
    retriever = RAGRetriever(
        vector_db_path=os.getenv("VDB_SAVE_PATH", "data/vector_db"),
        llm_api=llm_api,
        num_chunks=3
    )
    return retriever


def get_query_embedding(query, embedding_api):
    return embedding_api.generate_embedding(query)


def main():
    st.set_page_config(page_title="University Chatbot", page_icon="🎓", layout="centered")

    st.title("🎓 University Chatbot")
    st.caption("Ask me anything about the university!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_db_built" not in st.session_state:
        st.session_state.vector_db_built = False

    with st.sidebar:
        st.header("Settings")

        if st.button("Build/Rebuild Vector DB"):
            with st.spinner("Building vector database..."):
                try:
                    result = build_vector_db()
                    st.session_state.vector_db_built = True
                    st.success(f"Vector DB built! {result['num_chunks']} chunks created.")
                except Exception as e:
                    st.error(f"Error building vector DB: {str(e)}")

        st.divider()
        st.info("Type your question in the chat below!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if not st.session_state.vector_db_built:
                st.warning("Please build the vector database first using the sidebar button.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        retriever = load_retriever()
                        embedding_api = EmbeddingAPI()

                        query_embedding = get_query_embedding(query, embedding_api)
                        response = retriever.generate_response(query, query_embedding)

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
