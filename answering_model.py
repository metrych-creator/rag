from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sentence_transformers import CrossEncoder
from vector_stores import load_faiss, search_faiss
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from google import genai
from langfuse import get_client, observe
from langchain_google_vertexai import ChatVertexAI
from ragas.llms import LangchainLLMWrapper
import google.auth
from google.genai import types
from rank_bm25 import BM25Okapi

load_dotenv() 
google_api_key = os.getenv("GOOGLE_API_KEY")
pdf_path = "data/ifc-annual-report-2024-financials.pdf"


def reciprocal_rank_fusion(faiss_results, bm25_results, k=100):
    """Combine rankings from two systems (FAISS i BM25) with RRF."""

    fused_scores = {}
    
    faiss_docs = [res['text'] for res in faiss_results]
    bm25_docs = [res['text'] for res in bm25_results]

    # ranking FAISS (semantic)
    for rank, doc in enumerate(faiss_docs):
        # Użyj dokumentu jako klucza
        if doc not in fused_scores:
            fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + 1 + k)

    # ranking BM25 (lexical)
    for rank, doc in enumerate(bm25_docs):
        if doc not in fused_scores:
            fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + 1 + k)

    sorted_docs = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return sorted_docs


@observe()
def answer_query_with_rag(query: str, answering_model, embedding_model_name='thenlper/gte-small', rerank=False, top_k=100, final_context_k_rerank=5, hybrid_serach=False, metadata_filter=None):
    # 1. RETRIEVAL
    faiss_store, pdf_texts = load_faiss(pdf_path, embedding_model_name)
    faiss_results = search_faiss(faiss_store, query, top_k=top_k, metadata_filter=metadata_filter)

    # 1A. SEMANTIC SEARCH - FAISS
    retrieved_docs = [res['text'] for res in faiss_results]
    final_context_docs = []

    if hybrid_serach:
        # 1B. LEXICAL (BM25)
        tokenized_corpus = [doc.split(" ") for doc in pdf_texts] 
        bm25 = BM25Okapi(tokenized_corpus)

        # tokenize query
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)

        # bm25 ranking
        bm25_ranking_indices = np.argsort(bm25_scores)[::-1]
        bm25_results = [{'text': pdf_texts[i]} for i in bm25_ranking_indices[:top_k]]

        # 1C. CONECTING DATA
        retrieved_docs = reciprocal_rank_fusion(faiss_results, bm25_results)

    top_n_docs = retrieved_docs[:100]

    # 2. RERANKING
    if rerank:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        query_doc_pairs = [[query, doc] for doc in top_n_docs]
        rerank_scores = reranker.predict(query_doc_pairs)
        scored_documents = list(zip(top_n_docs, rerank_scores))
        # sort
        reranked_documents_with_scores = sorted(
            scored_documents,
            key=lambda x: x[1],
            reverse=True)
        final_results = [doc for doc, score in reranked_documents_with_scores]
        final_context_docs = final_results
    else:
        final_context_docs = top_n_docs[:final_context_k_rerank]

    # 3. CONTEXT FOR LLM 
    context = "\n\n".join(final_context_docs)

    # 4. GENERATION
    agent_input = (
        f"""Context:
        {context}

        Question:
        {query}

        You have access to a retrieved context from a pdf document.
        Answer the question ONLY based on the context provided. If you can't answer then write: "No information in given context."
        """
    )

    agent = create_agent(answering_model, system_prompt=agent_input)

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        final_msg = event["messages"][-1]


    return final_msg.content, retrieved_docs


@observe()
def call_gemini(prompt: str, temperature: float = 0.7) -> str:
    load_dotenv() 
    client = genai.Client(
        vertexai=True,
        project='gd-gcp-internship-ds',
        location='global',
    )
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt, 
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )
    return response.text


@observe()
def create_llm_to_metric_evaluation(model_name: str):
    creds, _ = google.auth.default(quota_project_id='gd-gcp-internship-ds')
    llm = ChatVertexAI(
        model_name=model_name,
        credentials=creds,
        location='global',
        temperature=0
    )
    return LangchainLLMWrapper(llm)
