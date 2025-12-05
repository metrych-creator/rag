import ast
from tqdm.auto import tqdm
import pandas as pd
import json
from answering_model import call_gemini, answer_query_with_rag, create_llm_to_metric_evaluation
from get_pdf_data import get_pdf_as_document
from vector_stores import load_faiss
import random
import datasets
from langchain_core.vectorstores import VectorStore
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages.system import SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
import os
import glob
import matplotlib.pyplot as plt
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import evaluate
from ragas.evaluation import EvaluationDataset


pd.set_option("display.max_colwidth", 10)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""



def generate_qa_pairs(chunks, ifsave=True, n_samples=100, temperature=0.7):
    
    print(f"Generating {n_samples} QA couples...")

    outputs = []
    for sampled_context in tqdm(random.sample(chunks, n_samples)):
        context = sampled_context.page_content

        QA_generation_prompt = f"""
    Your task is to write a factoid question and an answer given a context.
    Your factoid question should be answerable with a specific, concise piece of factual information from the context.
    Your factoid question should be formulated in the same style as questions users could ask in a search engine.
    This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

    Provide your answer as follows:

    Output:::
    Factoid question: (your factoid question)
    Answer: (your answer to the factoid question)

    Now here is the context.

    Context: {context}\n
    Output:::"""


        # Generate QA couple
        output_QA_couple = call_gemini(
            QA_generation_prompt.format(context=context), temperature=temperature
        )

        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
            answer = output_QA_couple.split("Answer: ")[-1].strip()

            assert len(answer) < 300, "Answer is too long"
            outputs.append(
                {
                    "Question": question,
                    "Ground_Truth_Context": context,
                    "Ground_Truth_Answer": answer,
                    "Page_Number": sampled_context.metadata.get("page_number", "N/A"),
                    "Context_Content_Type": getattr(sampled_context, "content_type", "text")
                }
            )
        except:
            print("Failed to parse output, skipping...")
            print(sampled_context.page_content)
            continue


    # Save outputs
    if ifsave == True:
        df = pd.DataFrame(outputs)
        df.to_csv("data/generated_qa_pairs.csv", index=False, encoding="utf-8", header=True)



def select_eval_qa_pairs(qa_pairs):
    question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

    question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

    question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

    print("Generating critique for each QA couple...")
    for output in tqdm(qa_pairs):
        evaluations = {
            "groundedness": call_gemini(
                question_groundedness_critique_prompt.format(
                    context=output["Ground_Truth_Context"], question=output["Question"]
                ),
            ),
            "relevance": call_gemini(
                question_relevance_critique_prompt.format(question=output["Question"]),
            ),
            "standalone": call_gemini(
                question_standalone_critique_prompt.format(question=output["Question"]),
            ),
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except Exception as e:
            continue

    generated_questions = pd.DataFrame.from_dict(qa_pairs)
    generated_questions = generated_questions.loc[
        (generated_questions["groundedness_score"] >= 4)
        & (generated_questions["relevance_score"] >= 4)
        & (generated_questions["standalone_score"] >= 4)
    ]

    eval_dataset = datasets.Dataset.from_pandas(
        generated_questions, split="train", preserve_index=False
    )
    eval_dataset.to_csv("data/eval_dataset.json")



def run_rag_tests(
    eval_dataset: datasets.Dataset,
    answering_model,
    embedding_model_name,
    output_file: str,
    rerank: bool = False,
    hybrid: bool= False,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None, 
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset.to_dict(orient="records")):
        question = example["Question"]
        if question in [output["Question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_query_with_rag(question, answering_model, top_k=100, embedding_model_name=embedding_model_name, rerank=rerank, hybrid_serach=hybrid)

        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["Ground_Truth_Answer"]}')
        result = {
            "Question": question,
            "Ground_Truth_Answer": example["Ground_Truth_Answer"],
            "source_doc": example["Page_Number"],
            "Generated_Answer": answer,
            "Retrieved_Docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)



def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluator_name: str,
    evaluation_prompt_template: ChatPromptTemplate,
) -> None:
    """Evaluates generated answers. Modifies the given answer file in place for better checkpointing."""
    answers = []
    if os.path.isfile(answer_path):  # load previous generations if they exist
        answers = json.load(open(answer_path, "r"))

    for experiment in tqdm(answers):
        if f"eval_score_{evaluator_name}" in experiment:
            continue

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=experiment["Question"],
            response=experiment["Generated_Answer"],
            reference_answer=experiment["Ground_Truth_Answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        feedback, score = [
            item.strip() for item in eval_result.content.split("[RESULT]")
        ]
        experiment[f"eval_score_{evaluator_name}"] = score
        experiment[f"eval_feedback_{evaluator_name}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f)


def run_evaluation(eval_dataset, answering_model, eval_chat_model, embedding_model):
    READER_MODEL_NAME = "gemini-2.5-flash-lite"
    if not os.path.exists("./output"):
        os.mkdir("./output")

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
        ]
    )

    for chunk_size in [500, 1000, 2000]:
        for embedding_model_name in ["thenlper/gte-small", "sentence-transformers/all-MiniLM-L6-v2"]:
            for rerank in [False, True]:
                settings_name = f"chunk:{chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}_rerank:{rerank}_reader-model:{READER_MODEL_NAME}"
                output_file_name = f"./output/rag_{settings_name}.json"
                
                print(f"Running evaluation for {settings_name}:")
                print("Loading knowledge base embeddings...")
                knowledge_index = load_faiss('data/ifc-annual-report-2024-financials.pdf', embedding_model=embedding_model)

                print("Running RAG...")
                
                run_rag_tests(
                    eval_dataset=eval_dataset,
                    answering_model=answering_model,
                    embedding_model=embedding_model,
                    knowledge_index=knowledge_index,
                    output_file=output_file_name,
                    reranker=rerank,
                    verbose=False,
                    test_settings=settings_name,
                )

                print("Running evaluation...")
                evaluate_answers(
                    output_file_name,
                    eval_chat_model,
                    evaluation_prompt_template = evaluation_prompt_template,
                    evaluator_name = 'gemini',
                )
    outputs = []
    for file in glob.glob("./output/*.json"):
        output = pd.DataFrame(json.load(open(file, "r")))
        output["settings"] = file
        outputs.append(output)
    result = pd.concat(outputs)
    result["eval_score_gemini"] = result["eval_score_gemini"].apply(
    lambda x: int(x) if isinstance(x, str) else 1
    )
    result["eval_score_gemini"] = (result["eval_score_gemini"] - 1) / 4
    average_scores = result.groupby("settings")["eval_score_gemini"].mean()
    average_scores.sort_values()


def load_and_print_rag_results(output_folder="./output"):
    dfs = []
    for f in glob.glob(f"{output_folder}/*.json"):
        data = json.load(open(f))
        if data:
            df = pd.DataFrame(data)
            df["settings"] = f  # Add settings column
            dfs.append(df)
    if not dfs:
        return print("No results found.")

    df = pd.concat(dfs, ignore_index=True)
    df["eval_score_gemini"] = df["eval_score_gemini"].apply(lambda x: int(x) if isinstance(x, str) else 1)
    avg_scores = ((df["eval_score_gemini"] - 1)/4).groupby(df["settings"]).mean().sort_values(ascending=False)
    print("Average RAG Evaluation Scores:\n", avg_scores)
    return df, avg_scores


def load_plot_rag_results(output_folder="./output"):
    # Load and concatenate
    dfs = []
    for f in glob.glob(f"{output_folder}/*.json"):
        data = json.load(open(f))
        if data:
            df = pd.DataFrame(data)
            df["settings"] = f
            dfs.append(df)
    if not dfs:
        return print("No results found.")

    df = pd.concat(dfs, ignore_index=True)
    df["eval_score_gemini"] = df["eval_score_gemini"].apply(lambda x: int(x) if isinstance(x, str) else 1)
    avg_scores = ((df["eval_score_gemini"] - 1)/4).groupby(df["settings"]).mean()

    # Clean file names
    avg_scores.index = [x.replace(output_folder+"/", "").replace(".json", "") for x in avg_scores.index]

    # Plot
    n_bars = len(avg_scores)
    plt.figure(figsize=(12, max(6, n_bars*0.25)))
    avg_scores.sort_values(ascending=False).plot(kind="barh")
    plt.ylabel("Normalized eval_score_gemini")
    plt.xticks(rotation=45, ha="right")
    plt.title("Average RAG Evaluation Scores")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig("rag_evaluation_scores.png")
    plt.show()
    plt.close()
    return df, avg_scores


def llm_evaluate_rag_models():
    pdf_path = "data/ifc-annual-report-2024-financials.pdf"
    chunks = get_pdf_as_document(pdf_path=pdf_path)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    answering_model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    eval_chat_model = init_chat_model("google_genai:gemini-2.5-flash-lite")

    # prepare evaluation dataset
    if not os.path.exists("data/eval_dataset.csv"):
        generate_qa_pairs(chunks, ifsave=True, temperature=0.7, n_samples=100)
        generated_qa_pairs = pd.read_csv("data/generated_qa_pairs.csv", encoding="utf-8", header=0)
        rag_evaluation_questions = pd.read_csv("data/RAG_evaluation.csv", encoding="utf-8", header=0)
        df_combined = pd.concat([generated_qa_pairs, rag_evaluation_questions], ignore_index=True)
        df_combined.to_csv("data/combined_qa_pairs.csv", index=False, encoding="utf-8", header=True)

    if not os.path.exists("data/combined_qa_pairs.csv"):
        combined_qa_pairs = pd.read_csv("data/combined_qa_pairs.csv", encoding="utf-8", header=0)
        combined_qa_pairs = combined_qa_pairs[combined_qa_pairs['Context_Content_Type'] == 'text']
        select_eval_qa_pairs(combined_qa_pairs.to_dict(orient="records"))

    json_files = glob.glob(f"./output/*.json")
    if not os.path.exists("./output") or len(json_files) == 0:
        eval_dataset = pd.read_csv("data/eval_dataset.csv", encoding="utf-8", header=0)
        run_evaluation(eval_dataset, answering_model, eval_chat_model, embedding_model)

    load_and_print_rag_results()
    load_plot_rag_results()


def wrap_llm_output(raw_output):
    if isinstance(raw_output, dict) and "statements" in raw_output:
        return {"text": "\n".join([s["statement"] for s in raw_output["statements"]])}
    return raw_output


def metric_rag_evaluation(top_k=10, rerank=False, hybrid=False, metadana_filter=None):
    # check if dataset with model answer exists
    if not os.path.exists(f"data/RAG_evaluation_with_responses_topk_{top_k}_rerank_{rerank}_hybrid_{hybrid}.csv"):
        df = pd.read_csv("data/RAG_evaluation.csv")
        answering_model = init_chat_model("gemini-2.5-flash-lite")            
        
        results = df["Question"].apply(
            lambda q: answer_query_with_rag(q, answering_model, top_k=top_k, rerank=rerank, hybrid_serach=hybrid, metadata_filter=metadana_filter))
        
        df["response"] = results.apply(lambda x: x[0])
        df["retrieved_contexts"] = results.apply(lambda x: x[1])
        df.to_csv(f"data/RAG_evaluation_with_responses_topk_{top_k}_rerank_{rerank}_hybrid_{hybrid}.csv", index=False)
    else:
        df = pd.read_csv(f"data/RAG_evaluation_with_responses_topk_{top_k}_rerank_{rerank}_hybrid{hybrid}.csv")
        df["retrieved_contexts"] = df["retrieved_contexts"].apply(
            lambda x: x if isinstance(x, list) else ast.literal_eval(x))
    
    # prepare dataset to evaluation
    df = df.rename(columns={
    "Question": "user_input",
    "Ground_Truth_Answer": "reference",
    'Context_Content_Type': 'Context_Content_Type'
    })

    evaluation_dataset = EvaluationDataset.from_pandas(df)
    raw_evaluator_llm = create_llm_to_metric_evaluation("gemini-2.5-flash-lite")
    evaluator_llm = wrap_llm_output(raw_evaluator_llm)
    result = evaluate(dataset=evaluation_dataset, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()], llm=evaluator_llm)

    print(result)

    # save scores
    with open(f'output/metrics_top_k_{top_k}_rerank_{rerank}_hybrid{hybrid}.txt', 'w', encoding='utf-8') as f:
        f.write(str(result))