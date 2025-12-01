# Multimodal RAG (Retrieval-Augmented Generation)

Results for model without using RAG, Gemini 2.5-flash-lite answers in questions dataset.
{'context_recall': 0.9118, 'faithfulness': 0.3938, 'factual_correctness(mode=f1)': 0.2594}

Results for model with using RAG with Gemini as evaluator:
{'context_recall': 0.9118, 'faithfulness': 0.1720, 'factual_correctness(mode=f1)': 0.1509}

Results for only text queries:
{'context_recall': 1.0000, 'faithfulness': 0.4250, 'factual_correctness(mode=f1)': 0.3290}
