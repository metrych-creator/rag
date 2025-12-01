from langchain_huggingface import HuggingFaceEmbeddings
from get_pdf_data import get_pdf_as_document
from langchain_community.vectorstores import FAISS
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import time
import numpy as np


def load_faiss(pdf_path, embedding_model_name, faiss_path="vector_stores/faiss_index"):
    if os.path.exists(faiss_path):
        return FAISS.load_local(faiss_path, embedding_model_name, allow_dangerous_deserialization=True)
    else:
        docs = get_pdf_as_document(pdf_path)
        vector_store = FAISS.from_documents(docs, embedding_model_name, distance_metric="cosine")
        vector_store.save_local(faiss_path)
        return vector_store
    

def ensure_qdrant_collection(client, collection_name, embedding_dim):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance="Cosine")
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


def search_faiss(vector_store, query, embedding_model_name, top_k=3):
    # Embed and normalize query
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    query_vec = np.array(embedding_model.embed_query(query))
    query_vec_norm = query_vec / np.linalg.norm(query_vec)

    # Search FAISS
    distances, indices = vector_store.index.search(query_vec.reshape(1, -1), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_id = vector_store.index_to_docstore_id[int(idx)]
        doc = vector_store.docstore.search(doc_id)

        # Reconstruct and normalize vector
        stored_vec = vector_store.index.reconstruct(int(idx))
        stored_vec_norm = stored_vec / np.linalg.norm(stored_vec)

        # Cosine similarity
        cosine = float(np.dot(query_vec_norm, stored_vec_norm))

        results.append({
            "text": doc.page_content,
            "score": cosine,
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
    faiss_results = search_faiss(faiss_store, query, embedding_model)
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


