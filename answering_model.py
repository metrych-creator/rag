from langchain_huggingface import HuggingFaceEmbeddings
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


load_dotenv() 
google_api_key = os.getenv("GOOGLE_API_KEY")
pdf_path = "data/ifc-annual-report-2024-financials.pdf"


@observe()
def answer_query_with_rag(query: str, answering_model, embedding_model_name='thenlper/gte-small', rerank=None):
    faiss_store = load_faiss(pdf_path, embedding_model_name)
    faiss_results = search_faiss(faiss_store, query, embedding_model_name, top_k=10)

    texts = [res['text'] for res in faiss_results]
    context = "\n\n".join(texts)

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


    return final_msg.content, texts


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
