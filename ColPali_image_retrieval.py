import os
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv() 
google_api_key = os.getenv("GOOGLE_API_KEY")

def convert_pdfs_to_images(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    all_images = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        all_images[doc_id] = images

    return all_images


def create_image_embeddings():
    index_path = ".byaldi/image_index/"
    if not (os.path.exists(index_path) and os.listdir(index_path)):
        docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device = "cpu")
        docs_retrieval_model.index(
            input_path="data/pdfs",
            index_name="image_index",
            overwrite=True
        )
    else:
        # docs_retrieval_model = RAGMultiModalModel.from_index(index_path)
        docs_retrieval_model = RAGMultiModalModel.from_index("./image_index", device='cpu')

    return docs_retrieval_model


def get_grouped_images(results, all_images):
    grouped_images = []

    for result in results:
        doc_id = result['doc_id']
        page_num = result['page_num']
        grouped_images.append(all_images[doc_id][page_num - 1]) # page_num are 1-indexed, while doc_ids are 0-indexed.

    return grouped_images


def answer_with_images(query: str, grouped_images):
    system_instruction = (
        """You are a RAG system. Your task is to analyze the attached images (document pages) 
        in order to find and provide a precise answer to the user's question. 
        If the information is directly in the table or chart, use it.
        Answer concisely and cite the page from which the information comes."""
    )
    full_contents = [system_instruction]
    full_contents.extend(grouped_images)
    full_contents.append(f"This is original query: {query}")

    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_contents
    )
    return response.text





def answer_query_with_images(query: str, top_k: int=10):
    all_pages = convert_pdfs_to_images("data/pdfs")
    docs_retrieval_model = create_image_embeddings()
    results = docs_retrieval_model.search(query, k=top_k)
    grouped_images = get_grouped_images(results, all_pages)
    return answer_with_images(query, grouped_images)
     