"""
Evaluation script for the RAG-from-scratch assignment.

Run this file from the same directory as rag_pipeline.py.
"""

import importlib
import json
import sys
import textwrap
import time
from typing import Callable


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"


def c(text, colour):
    return f"{colour}{text}{RESET}"


def header(title):
    print(f"\n{BOLD}{CYAN}{'-' * 60}\n  {title}\n{'-' * 60}{RESET}")


def rule():
    print(c("-" * 60, DIM))


QUESTIONS = [
    {
        "id": "A1",
        "category": "retrieval",
        "query": "Which city in the documents served as Japan's imperial capital for more than a thousand years?",
        "keywords": ["Kyoto", "imperial capital"],
    },
    {
        "id": "A2",
        "category": "retrieval",
        "query": "Who is often called the first computer programmer in the documents?",
        "keywords": ["Lovelace", "first computer programmer"],
    },
    {
        "id": "A3",
        "category": "retrieval",
        "query": "What date is given for the Apollo 11 launch?",
        "keywords": ["July 16, 1969"],
    },
    {
        "id": "A4",
        "category": "retrieval",
        "query": "What process explains the large-scale movement of Earth's lithosphere?",
        "keywords": ["Plate tectonics"],
    },
    {
        "id": "A5",
        "category": "retrieval",
        "query": "What process in plants converts light energy into chemical energy?",
        "keywords": ["Photosynthesis"],
    },
    {
        "id": "B1",
        "category": "correctness",
        "query": "What idea did Ada Lovelace propose about what a general-purpose machine could manipulate?",
        "keywords": [],
    },
    {
        "id": "B2",
        "category": "correctness",
        "query": "Where does the light-dependent reaction occur?",
        "keywords": [],
    },
    {
        "id": "B3",
        "category": "correctness",
        "query": "Explain the relationship between the light-dependent reactions and the Calvin cycle.",
        "keywords": [],
    },
    {
        "id": "B4",
        "category": "correctness",
        "query": "What made Apollo 11 a major technological achievement according to the documents?",
        "keywords": [],
    },
    {
        "id": "B5",
        "category": "correctness",
        "query": "What happens at divergent boundaries according to the documents?",
        "keywords": [],
    },
    {
        "id": "C1",
        "category": "grounding",
        "query": "What was Ada Lovelace's favorite piece of music?",
        "keywords": [],
    },
    {
        "id": "C2",
        "category": "grounding",
        "query": "List five facts about Kyoto that are not included in the documents.",
        "keywords": [],
    },
    {
        "id": "C3",
        "category": "grounding",
        "query": "What is the current population of Kyoto?",
        "keywords": [],
    },
    {
        "id": "C4",
        "category": "grounding",
        "query": "What did experts say about plate tectonics in 2024?",
        "keywords": [],
    },
    {
        "id": "C5",
        "category": "grounding",
        "query": "Invent a plausible next chapter for the Apollo 11 mission after splashdown.",
        "keywords": [],
    },
]


def load_pipeline():
    try:
        pipeline = importlib.import_module("rag_pipeline")
    except ModuleNotFoundError:
        print(c("\n[ERROR] Could not import rag_pipeline.py", RED))
        sys.exit(1)

    missing = []
    if not hasattr(pipeline, "retrieve"):
        missing.append("retrieve(query, k=3)")
    if not hasattr(pipeline, "generate"):
        missing.append("generate(query, chunks)")
    if missing:
        print(c(f"\n[ERROR] rag_pipeline.py is missing: {', '.join(missing)}", RED))
        sys.exit(1)

    print(c("  rag_pipeline.py loaded successfully.", GREEN))
    return pipeline.retrieve, pipeline.generate


def auto_precision(chunks: list[str], keywords: list[str]):
    if not keywords:
        return None
    joined = " ".join(chunks).lower()
    hit = all(keyword.lower() in joined for keyword in keywords)
    return 1.0 if hit else 0.0


def ask_human(question_id: str, query: str, chunks: list[str], answer: str, category: str):
    rule()
    print(f"  {BOLD}Question {question_id}{RESET}  [{c(category.upper(), YELLOW)}]")
    print(f"  {c('Query:', DIM)} {query}\n")

    print(f"  {c('Retrieved chunks:', DIM)}")
    for idx, chunk in enumerate(chunks, start=1):
        wrapped = textwrap.fill(chunk.strip(), width=56, initial_indent="    ", subsequent_indent="    ")
        print(f"  {c(f'Chunk {idx}:', DIM)}\n{wrapped}\n")

    print(f"  {c('Generated answer:', DIM)}")
    wrapped_answer = textwrap.fill(answer.strip(), width=56, initial_indent="    ", subsequent_indent="    ")
    print(f"{wrapped_answer}\n")

    if category == "grounding":
        prompt = "  Does the answer stay within the retrieved chunks (no hallucination)? [1=yes / 0=no]: "
    else:
        prompt = "  Is the answer factually correct? [1=yes / 0=no / s=skip]: "

    while True:
        score = input(prompt).strip().lower()
        if score in ("0", "1"):
            return int(score)
        if score == "s":
            return None
        print(c("  Please enter 1, 0, or s.", RED))


def run_eval(retrieve: Callable, generate: Callable):
    results = []
    retrieval_scores = []
    correctness_scores = []
    grounding_scores = []

    total = len(QUESTIONS)
    for idx, question in enumerate(QUESTIONS, start=1):
        header(f"Question {question['id']}  ({idx}/{total})")

        t0 = time.time()
        try:
            chunks = retrieve(question["query"], k=3)
        except Exception as error:
            print(c(f"  [ERROR] retrieve() raised: {error}", RED))
            chunks = []
        retrieval_time = round(time.time() - t0, 3)

        if not chunks:
            print(c("  No chunks returned - skipping generation.", YELLOW))
            results.append(
                {
                    **question,
                    "chunks_retrieved": [],
                    "answer": "",
                    "auto_retrieval_score": 0,
                    "human_score": None,
                    "retrieval_time_s": retrieval_time,
                    "generation_time_s": 0,
                }
            )
            retrieval_scores.append(0)
            continue

        t1 = time.time()
        try:
            answer = generate(question["query"], chunks)
        except Exception as error:
            print(c(f"  [ERROR] generate() raised: {error}", RED))
            answer = ""
        generation_time = round(time.time() - t1, 3)

        print(c(f"  Retrieval: {retrieval_time}s  |  Generation: {generation_time}s", DIM))
        auto_score = auto_precision(chunks, question.get("keywords", []))

        if question["category"] == "retrieval" and auto_score is not None:
            human_score = None
            retrieval_scores.append(auto_score)
            status = c("AUTO: PASS", GREEN) if auto_score else c("AUTO: FAIL", RED)
            print(f"\n  {status}  (keyword match on retrieved chunks)")
        else:
            human_score = ask_human(question["id"], question["query"], chunks, answer, question["category"])
            if question["category"] == "retrieval":
                retrieval_scores.append(human_score if human_score is not None else 0)
            elif question["category"] == "correctness":
                correctness_scores.append(human_score if human_score is not None else 0)
            elif question["category"] == "grounding":
                grounding_scores.append(human_score if human_score is not None else 0)

        results.append(
            {
                "id": question["id"],
                "category": question["category"],
                "query": question["query"],
                "chunks_retrieved": chunks,
                "answer": answer,
                "auto_retrieval_score": auto_score,
                "human_score": human_score,
                "retrieval_time_s": retrieval_time,
                "generation_time_s": generation_time,
            }
        )

    return results, retrieval_scores, correctness_scores, grounding_scores


def print_report(results, retrieval_scores, correctness_scores, grounding_scores):
    def safe_mean(scores):
        filtered = [score for score in scores if score is not None]
        return round(sum(filtered) / len(filtered), 3) if filtered else 0.0

    retrieval_mean = safe_mean(retrieval_scores)
    correctness_mean = safe_mean(correctness_scores)
    grounding_mean = safe_mean(grounding_scores)
    overall = safe_mean([retrieval_mean, correctness_mean, grounding_mean])

    thresholds = {
        "Retrieval Precision@3": (retrieval_mean, 0.60),
        "Answer Correctness": (correctness_mean, 0.60),
        "Grounding Score": (grounding_mean, 0.70),
    }

    header("FINAL EVALUATION REPORT")
    for label, (score, threshold) in thresholds.items():
        status = c("PASS", GREEN) if score >= threshold else c("FAIL", RED)
        bar_length = int(score * 20)
        bar = c("#" * bar_length, GREEN if score >= threshold else RED) + c("." * (20 - bar_length), DIM)
        print(f"  {label:<28} {bar}  {score:.2f}  [{status}] (min {threshold})")

    rule()
    print(f"  {'Overall':28} {' ' * 22} {overall:.2f}")
    rule()

    return {
        "retrieval_precision_at_3": retrieval_mean,
        "answer_correctness": correctness_mean,
        "grounding_score": grounding_mean,
        "overall": overall,
        "pass_threshold": {
            "retrieval": 0.60,
            "correctness": 0.60,
            "grounding": 0.70,
        },
    }


def main():
    print(f"\n{BOLD}{'=' * 60}")
    print("  RAG EVALUATION SCRIPT")
    print(f"{'=' * 60}{RESET}")
    print(c("  Importing rag_pipeline.py ...", DIM))

    retrieve, generate = load_pipeline()
    print(c(f"\n  {len(QUESTIONS)} questions loaded across 3 categories.\n", DIM))

    input(c("  Press ENTER to begin evaluation ...", YELLOW))

    results, retrieval_scores, correctness_scores, grounding_scores = run_eval(retrieve, generate)
    summary = print_report(results, retrieval_scores, correctness_scores, grounding_scores)

    output = {"summary": summary, "results": results}
    out_path = "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(c(f"\n  Results saved to {out_path}\n", DIM))


if __name__ == "__main__":
    main()
