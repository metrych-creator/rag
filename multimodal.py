import os 
from pypdf import PdfReader
from tabula import read_pdf
import fitz
from unstructured.partition.pdf import partition_pdf
from typing import Any
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image 
import pandas as pd
from joblib import Parallel, delayed
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import regex as re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import uuid

TABLE_FOLDER = "tables"
IMAGE_FOLDER = "extracted_images"
FAISS_SAVE_PATH = "vector_stores/multimodal/faiss_index"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv() 
google_api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client()

CHAT_MODEL = "gemini-2.5-flash-lite"
HF_EMBEDDING_MODEL_NAME = 'thenlper/gte-small' 

PREAMBLE_EN = """## Task & Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."""

PROMPT_TEMPLATE = """You are an assistant tasked with summarizing tables, images and text. \
Give a concise summary of the table, image or text. Table, image or text chunk: {element}. Only provide the summary and no other text."""
SUMMARY_PREAMBLE = "You are an expert summarization assistant. Your task is to provide a concise summary of the provided text or table chunk. Your output must contain only the summary and no introductory phrases."


class Element(BaseModel):
    type: str
    text: Any


def extract_tabular_data(pdf_path):
    PAGES = len(PdfReader(pdf_path).pages)

    for page in range(1, PAGES + 1):
        tables_on_page = read_pdf(pdf_path, pages=page, pandas_options={'header': None})
        
        for i, t in enumerate(tables_on_page):
            df = pd.DataFrame(t)
            file_name = os.path.join(TABLE_FOLDER, f"table_p{page}_{i+1}.csv")
            df.to_csv(file_name, index=False, header=1)


def extract_images(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc[page_index].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            with open(f"{output_folder}/image_{page_index}_{img_index}.{image_ext}", "wb") as f:
                f.write(image_bytes)


def extract_texts(pdf_path):
    texts = partition_pdf(
        filename=pdf_path,
        strategy="auto", 
        languages=['English']
    )

    return [str(t) for t in texts]


def get_chat_output(message: str, preamble: str, chat_history: list, model: str, temp: float, documents: list = None, images: list = None, csv_files: list = None) -> str:
    """
    Generates a response from the Gemini model, handling history, RAG context,
    and optional images in the new user message.
    """
    
    # 1. Building Context (RAG)
    context = message
    if documents:
        document_texts = "\n".join([f"## Document {i+1}\nTitle: {doc.get('title', 'N/A')}\nContent: {doc.get('snippet', '')}" for i, doc in enumerate(documents)])
        context = f"Answer the question strictly based on the documents provided below:\n{document_texts}\n\nQuestion: {message}"
    
    # 2. Converting Chat History
    history_converted = []
    chat_history = chat_history if chat_history is not None else []

    for msg in chat_history:
        role = "user" if msg['role'].upper() == "USER" else "model"
        history_converted.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['message'])]))

    # 3. Building the NEW Content for the current message (text + images)
    user_parts = []

    # Adding images
    if images and isinstance(images, list):
        for image_path in images:
            if os.path.exists(image_path):
                try:
                    # Determine MIME type based on extension
                    mime_type = 'image/jpeg'
                    if image_path.lower().endswith(('.png', '.webp')):
                         mime_type = 'image/png'
                    elif image_path.lower().endswith(('.gif')):
                         mime_type = 'image/gif'
                    
                    # Load binary image data
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                        
                    user_parts.append(types.Part.from_bytes(
                        data=image_data,
                        mime_type=mime_type
                    ))
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
            else:
                 print(f"Warning: File does not exist at path: {image_path}")
    
    # Adding CSV Files
    if csv_files and isinstance(csv_files, list):
        for csv_path in csv_files:
            if os.path.exists(csv_path):
                try:
                    # MIME type for CSV files is 'text/csv'
                    mime_type = 'text/csv'
                    
                    # Load CSV data as bytes
                    with open(csv_path, 'rb') as f:
                        csv_data = f.read()
                        
                    user_parts.append(types.Part.from_bytes(
                        data=csv_data,
                        mime_type=mime_type
                    ))
                except Exception as e:
                    print(f"Error loading CSV file {csv_path}: {e}")
            else:
                 print(f"Warning: CSV file does not exist at path: {csv_path}")

    # Adding the query text (context/message)
    final_context_str = str(context)
    user_parts.append(types.Part.from_text(text=final_context_str))
    
    # Adding the complete, multimodal user message to history
    history_converted.append(types.Content(role="user", parts=user_parts))

    # 4. Configuration and API Call
    config = types.GenerateContentConfig(
        temperature=temp,
        system_instruction=preamble
    )
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=history_converted,
            config=config,
        )
        return response.text
    except NameError:
         return "Error: Global variable 'client' has not been defined."
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Sorry, an error occurred while generating the response."


def parallel_proc_chat(prompts: list[str], preamble: str, chat_history: list = None, model: str = CHAT_MODEL, temp: float = 0.1, n_jobs: int = 10) -> list[str]:
    """Parallel processing of chat endpoint calls."""
        
    responses = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(get_chat_output)(
            message = prompt,
            preamble = preamble,
            chat_history = chat_history,
            model = CHAT_MODEL,
            temp = temp,
            documents=None,
            images = [prompt]
        ) for prompt in prompts
    )
    return responses


def rerank(query: str, returned_documents: list, top_n: int = 3) -> list[str]:
    # ??? it doesn't works in this way, how it should - it doesn't use similarity
    if len(returned_documents) > 0:
        return returned_documents[:top_n]
    else:
        return []


def parallel_proc_chat_dynamic(elements: list[str], preamble: str, model: str, temp: float, n_jobs: int = 10) -> list[str]:    
    def prepare_chat_call(element: str, preamble: str, model: str, temp: float):
        images = None
        csv_files = None
        
        if os.path.exists(element):
            file_extension = os.path.splitext(element)[1].lower()

            # images
            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                message = "Provide a summary of the attached image."
                images = [element]

            # tables
            elif file_extension == '.csv':
                message = "Analyze the attached CSV table and provide a concise summary of its contents and key findings."
                csv_files = [element]
            else:
                message = PROMPT_TEMPLATE.format(element=f"File path: {element} (Unsupported type)")
            return get_chat_output(message, preamble, [], model, temp, documents=None, images=images, csv_files=csv_files)
        else:
            message = PROMPT_TEMPLATE.format(element=element)
            return get_chat_output(message, preamble, [], model, temp, documents=None, images=None, csv_files=None)

    responses = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(prepare_chat_call)(
            element, 
            preamble, 
            model, 
            temp
        ) for element in elements
    )
    return responses


def build_multimodal_vectorstore(all_elements, all_summaries, texts, tables, images, doc_ids, embedding_function=HF_EMBEDDING_MODEL_NAME):
    all_documents_to_embed = []
    
    for i, (content, summary) in enumerate(zip(all_elements, all_summaries)):

        if i < len(texts):
            doc_type = "text"
        elif i < len(texts) + len(tables):
            doc_type = "table"
        else:
            doc_type = "image"
            
        enhanced_content = (
            f"TYPE: {doc_type}\n"
            f"SUMMARIZED CONTEXT: {summary}\n\n"
            f"FULL CONTENT: {content}"
        )
        
        doc = Document(
            page_content=enhanced_content, 
            metadata={
                "original_id": doc_ids[i],
                "doc_type": doc_type,
                "summary": summary
            }
        )
        all_documents_to_embed.append(doc)

    vectorstore = FAISS.from_documents(
        documents=all_documents_to_embed,
        embedding=embedding_function,
    )

    retriever = vectorstore.as_retriever()

    vectorstore.save_local(FAISS_SAVE_PATH)
    
    return vectorstore, retriever


def load_faiss_vectorstore(faiss_path: str = FAISS_SAVE_PATH, embedding_function = HF_EMBEDDING_MODEL_NAME):
    # ??? probably i have it
    vectorstore = FAISS.load_local(
        faiss_path, 
        embedding_function = embedding_function, 
        allow_dangerous_deserialization=True 
    )
    retriever = vectorstore.as_retriever()
    return vectorstore, retriever


def prepare_multimodal_data_and_vectorstore(pdf_path):
    categorized_elements = []
    os.makedirs(TABLE_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    # tables
    if len(os.listdir(TABLE_FOLDER)) == 0:
        extract_tabular_data(pdf_path)
        print('tables extracted')

    # images
    if len(os.listdir(IMAGE_FOLDER)) == 0:
        extract_images(pdf_path, IMAGE_FOLDER)  
        print('images extracted')
  

    texts = extract_texts(pdf_path)
    print('texts extracted')

    tables = os.listdir(TABLE_FOLDER)
    images = os.listdir(IMAGE_FOLDER)

    print(f"""Extracted:\n
          tables: {len(texts)},\n
          images: {len(tables)},\n
          texts: {len(images)}""")
    

    all_elements = [*texts, *tables, *images]
    all_summaries = parallel_proc_chat_dynamic(all_elements, SUMMARY_PREAMBLE, CHAT_MODEL, 0.1)

    embedding_function = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL_NAME) 
    all_documents_to_embed = []
    doc_ids = [str(uuid.uuid4()) for _ in all_elements]

    # build vectorstore
    vectorstore, retriever = build_multimodal_vectorstore(all_elements, all_summaries, texts, tables, images, doc_ids=doc_ids)
    print('vectorstore created')
    return vectorstore, retriever


def process_query(query: str, retriever) -> tuple[str, str]:
    """Runs retrieval, rerank, and final generation in one call (RAG)."""
    # --- Step 1: Search ---
    docs = retriever.invoke(query) 
    doc_texts = [d.page_content for d in docs]
    
    reranked_docs = rerank(query, doc_texts) 
    
    # formatting
    documents = [{"title": f"chunk {i}", "snippet": reranked_docs[i]} for i in range(len(reranked_docs))]
    
    # --- Step 2: Answer generation ---
    preamble = "You help people answer their questions based on the provided documents. Answer in full sentences."
    final_answer = get_chat_output(
        message=query,
        preamble=preamble,
        chat_history=[], 
        model=CHAT_MODEL,
        temp=0.2,
        documents=documents
    )
    context_str = "\n---\n".join(reranked_docs)
    return final_answer, context_str


def run_multimodal():
    pdf_path = 'data/ifc-annual-report-2024-financials.pdf'

    if os.path.exists(FAISS_SAVE_PATH):
        vectorstore, retriever = load_faiss_vectorstore()
    else:
        vectorstore, retriever = prepare_multimodal_data_and_vectorstore(pdf_path)

    query = 'What is in the first page?'
    final_answer, final_answer_docs = process_query(query, retriever)
    print("ANSWER:", final_answer, '\n')
    print('DOCUMENTS:\n ', final_answer_docs)



run_multimodal()


    
