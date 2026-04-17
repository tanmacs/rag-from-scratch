# RAG From Scratch

This project implements a minimal Retrieval-Augmented Generation pipeline in plain Python without LangChain, LlamaIndex, Haystack, or any vector database.

## Included deliverables

- `rag_pipeline.py`: full end-to-end pipeline with separate loader, chunker, embedder, retriever, and generator components
- `documents/`: five source text files, each above 200 words
- `rag_eval.py`: evaluation script with 15 questions across retrieval, correctness, and grounding
- `eval_results.json`: saved output from the evaluation run
- `reflection.md`: written reflection on design choices and failure modes

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the demo pipeline

```powershell
.\.venv\Scripts\python.exe rag_pipeline.py
```

## Run the evaluation

```powershell
.\.venv\Scripts\python.exe rag_eval.py
```
