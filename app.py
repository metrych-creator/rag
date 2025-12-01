import os 
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from vector_stores import compare_vector_stores
from view_query_rag import show_view
from rag_evaluation import evaluate_rag_models

os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    answering_model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    # VECTOR STORE COMPARISON
    # compare_vector_stores(embedding_model)
    # RAG EVALUATION
    # evaluate_rag_models()
    show_view(answering_model)