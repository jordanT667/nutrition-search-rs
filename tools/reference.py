"""Python reference for the Rust port, using the retriever of nutrition-rag.

1. tests/fixtures/small.json: a 3-passage corpus, queries and the BM25 scores from
   rank_bm25 (committed, checked by `cargo test` in CI).
2. target/python_scores.json: the scores of every passage for the 50 evaluation questions on
   the real corpus (not committed: it needs passages.json; checked by `cargo test` when present).
3. The time Python takes for the same searches as `nutrisearch bench`.

Usage, from this folder, with nutrition-rag next to it and its requirements installed:
    python tools/reference.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parents[1]
RAG = HERE.parent / "nutrition-rag"
sys.path.insert(0, str(RAG))

from src.nutrirag.retrieve import Retriever, load_passages, tokens  # noqa: E402

SMALL = [
    {"id": "spf2019-p34-0", "doc": "spf2019", "page": 34,
     "text": "Deux fois par semaine du poisson, dont un poisson gras (sardines, maquereau, hareng, saumon)."},
    {"id": "spf2019-p36-0", "doc": "spf2019", "page": 36,
     "text": "Limiter la charcuterie à 150 g par semaine, environ 3 tranches de jambon blanc."},
    {"id": "spf2019-p35-0", "doc": "spf2019", "page": 35,
     "text": "L'eau est la seule boisson recommandée. Les sodas, même light, doivent être limités."},
]
SMALL_QUERIES = ["combien de tranches de jambon ?", "poisson gras sardines", "Quelle boisson ? Les sodas light",
                 "semaine", "poisson poisson", "œufs et crème brûlée"]


def scores(retriever, queries):
    return [{"query": q, "tokens": tokens(q), "scores": retriever.scores(q, "bm25").tolist()} for q in queries]


def main():
    (HERE / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    small = {"passages": SMALL, "queries": scores(Retriever(SMALL), SMALL_QUERIES)}
    (HERE / "tests" / "fixtures" / "small.json").write_text(json.dumps(small, ensure_ascii=False, indent=1), encoding="utf-8")

    questions = [json.loads(line)["question"] for line in (HERE / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    passages = load_passages()
    t0 = time.perf_counter()
    BM25Okapi([tokens(p["text"]) for p in passages])  # same work as the Rust index: tokenize + BM25
    build = time.perf_counter() - t0
    retriever = Retriever(passages)
    (HERE / "target").mkdir(exist_ok=True)
    (HERE / "target" / "python_scores.json").write_text(json.dumps(scores(retriever, questions)), encoding="utf-8")

    rounds = 20
    t0 = time.perf_counter()
    for _ in range(rounds):
        for q in questions:
            s = retriever.bm25.get_scores(tokens(q))
            np.argsort(-s)[:5]
    secs = time.perf_counter() - t0
    n = rounds * len(questions)
    print(f"Python: BM25 index built in {build * 1e3:.2f} ms, "
          f"{n} searches in {secs:.3f} s: {n / secs:.0f} queries/s, {secs / n * 1e6:.1f} µs per query")


if __name__ == "__main__":
    main()
