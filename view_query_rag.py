from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

from answering_model import answer_query_with_rag

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"


def show_view(answering_model):
    st.set_page_config(layout="wide")
    st.header("RAG")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_query = st.chat_input("Ask a question about your document...")
    print("User query:", user_query)

    if user_query:
        st.session_state.messages.append({"role": "user", "type": "TEXT", "text": user_query})
        try:
            response, texts = answer_query_with_rag(user_query, answering_model, embedding_model_name)
            print("Response:", response)
        except Exception as e:
            if "503" in str(e) or 'UNAVAILABLE' in str(e):
                st.warning("The AI model is overloaded. Please try again in a few seconds.")
            else:
                st.error(e)
            return
        
        results_message = {"role": "assistant", "type": "TEXT", "text": response}
        st.session_state.messages.append(results_message)
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "TEXT":
                st.write(message["text"])
