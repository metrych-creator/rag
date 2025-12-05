import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from get_pdf_data import get_pdf_as_document
from langchain_community.vectorstores import FAISS
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import time
import numpy as np


def load_faiss(pdf_path, embedding_model_name='thenlper/gte-small', faiss_path="vector_stores/faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    texts_path = faiss_path + "_texts.pkl"

    if os.path.exists(faiss_path):
        with open(texts_path, 'rb') as f:
                pdf_texts = pickle.load(f)
                
        vector_store = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
        return vector_store, pdf_texts
    else:
        docs = get_pdf_as_document(pdf_path)
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(faiss_path)

        pdf_texts = [doc.page_content for doc in docs]
        with open(texts_path, 'wb') as f:
            pickle.dump(pdf_texts, f)
        return vector_store, pdf_texts
    

def ensure_qdrant_collection(client, collection_name, embedding_dim):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance="cosine")
    )


def load_qdrant(collection_name):
    client = QdrantClient(url="http://localhost:6333")
    ensure_qdrant_collection(client, collection_name, embedding_dim=384)
    return client


def add_emb_to_qdrant(client, docs, embeddings, collection_name):
    points = []
    for i, doc in enumerate(docs):
        vec = embeddings.embed_query(doc.page_content)
        points.append(
            PointStruct(
                id=i,
                vector=vec,
                payload={
                    "page_number": doc.metadata["page_number"],
                    "text": doc.page_content
                }
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def search_faiss(vector_store, query: str, top_k=10, metadata_filter=None):
    results_with_score = vector_store.similarity_search_with_score(
        query=query, 
        k=top_k, 
        filter=metadata_filter
    )

    results = []
    for doc, score in results_with_score:
        results.append({
            "text": doc.page_content,
            "score": score,
            "metadata": doc.metadata
        })

    return results


def search_qdrant(client, query, embedding_model, collection_name):
    query_vec = embedding_model.embed_query(query)
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=3
    ).points
    return results


def compare_vector_stores(embedding_model):
    pdf_path="data/ifc-annual-report-2024-financials.pdf"
    collection_name="ifc_reports_collection"

    query = "\nWhat was the incomes IFC in 2024?"
    print(query)

    faiss_store = load_faiss(embedding_model, pdf_path)

    docs = get_pdf_as_document(pdf_path)

    qdrant = load_qdrant(collection_name)
    add_emb_to_qdrant(qdrant, docs, embedding_model, collection_name)

    print(f"Docs number: {len(docs)}")


    print("\n========== FAISS RESULTS ==========")
    start = time.time()
    faiss_results = search_faiss(faiss_store, query)
    print(f"FAISS latency: {(time.time() - start):.4f} s")

    for i, res in enumerate(faiss_results, 1):
        print(f"\nFAISS #{i} (score {res["score"]:}")
        print(res["text"][:300], "...")


    print("\n========== QDRANT RESULTS ==========")
    start = time.time()
    qdrant_results = search_qdrant(qdrant, query, embedding_model, collection_name)
    print(f"Qdrant latency: {(time.time() - start):.4f} s")

    for i, res in enumerate(qdrant_results, 1):
        print(f"\nQdrant #{i} (score {res.score}):")
        print(res.payload["text"][:300], "...")


