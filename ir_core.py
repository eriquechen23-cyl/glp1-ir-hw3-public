"""Transparent sparse lexical retrieval for the GLP-1 homework corpus."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse


_GLP = re.compile(r"\b(?:glucagon[\s-]+like[\s-]+peptide[\s-]*1|glp[\s-]*1)\b", re.I)
_TOKEN = re.compile(r"[a-z][a-z0-9]*|[0-9]+[a-z][a-z0-9]*")


def tokenize(text: str) -> list[str]:
    """Lowercase scientific tokens without stemming or stopword removal.

    GLP-1 spellings collapse to glp1; a letter-number hyphen joins (COVID-19),
    while ordinary hyphens separate words. Standalone numbers are discarded.
    """
    text = _GLP.sub("glp1", str(text).lower())
    text = re.sub(r"(?<=[a-z])-(?=\d)", "", text)
    return _TOKEN.findall(text)


def load_documents(path: str | Path = "data/glp1_1000.jsonl") -> list[dict]:
    docs: list[dict] = []
    pmids: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pmid = str(row.get("pmid", "")).strip()
            if not pmid:
                raise ValueError(f"missing pmid at line {line_no}")
            if pmid in pmids:
                raise ValueError(f"duplicate pmid {pmid} at line {line_no}")
            if not (str(row.get("title") or "").strip() or str(row.get("abstract") or "").strip()):
                raise ValueError(f"empty title and abstract at line {line_no}")
            row["pmid"] = pmid
            pmids.add(pmid)
            docs.append(row)
    return docs


def _weighted_counts(counts: sparse.csr_matrix, method: str, idf: np.ndarray) -> sparse.csr_matrix:
    if method not in {"tf", "log_tf", "idf", "tfidf", "log_tfidf"}:
        raise ValueError(f"unknown weighting method: {method}")
    result = counts.astype(float).copy().tocsr()
    if method == "idf":
        result.data[:] = 1
    elif method in {"log_tf", "log_tfidf"}:
        result.data[:] = 1 + np.log(result.data)
    if method in {"idf", "tfidf", "log_tfidf"}:
        result = result.multiply(idf).tocsr()
    return result


def _l2(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    lengths = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inv = np.zeros_like(lengths)
    np.divide(1.0, lengths, out=inv, where=lengths > 0)
    return sparse.diags(inv).dot(matrix).tocsr()


class IRIndex:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.tokens = [tokenize(f"{row.get('title') or ''} {row.get('abstract') or ''}") for row in docs]
        self.terms = np.asarray(sorted({term for tokens in self.tokens for term in tokens}), dtype=str)
        self.vocabulary = {term: i for i, term in enumerate(self.terms)}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[int] = []
        for row_idx, tokens in enumerate(self.tokens):
            for term, count in Counter(tokens).items():
                rows.append(row_idx)
                cols.append(self.vocabulary[term])
                vals.append(count)
        self.counts = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.terms)), dtype=np.int32)
        self.df = np.asarray((self.counts > 0).sum(axis=0)).ravel().astype(int)
        self.idf = np.log((len(docs) + 1) / (self.df + 1)) + 1
        self._matrices: dict[tuple[str, bool], sparse.csr_matrix] = {}

    def matrix(self, method: str, l2: bool = False) -> sparse.csr_matrix:
        key = (method, l2)
        if key not in self._matrices:
            mat = _weighted_counts(self.counts, method, self.idf)
            self._matrices[key] = _l2(mat) if l2 else mat
        return self._matrices[key]

    def query_vector(self, query: str, method: str, l2: bool = False) -> sparse.csr_matrix:
        found = Counter(t for t in tokenize(query) if t in self.vocabulary)
        cols = [self.vocabulary[t] for t in found]
        raw = sparse.csr_matrix((list(found.values()), ([0] * len(cols), cols)), shape=(1, len(self.terms)), dtype=np.int32)
        weighted = _weighted_counts(raw, method, self.idf)
        return _l2(weighted) if l2 else weighted

    def rank(self, query: str, method: str = "log_tfidf", metric: str = "cosine", limit: int | None = 10) -> list[dict]:
        if metric not in {"cosine", "dot"}:
            raise ValueError(f"unknown metric: {metric}")
        if metric == "cosine":
            q = self.query_vector(query, method, l2=True)
            docs = self.matrix(method, l2=True)
        else:
            q = self.query_vector(query, method)
            docs = self.matrix(method)
        if q.nnz == 0:
            return []
        scores = np.asarray(docs.dot(q.T).toarray()).ravel()
        positive = np.flatnonzero(scores > 0)
        ordered = sorted(positive, key=lambda i: (-scores[i], int(i)))
        if limit is not None:
            ordered = ordered[:limit]
        return [{"doc_index": int(i), "pmid": str(self.docs[i]["pmid"]),
                 "title": str(self.docs[i].get("title") or ""), "score": float(scores[i])} for i in ordered]


def cosine(a, b) -> float | None:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("vectors have different dimensions")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else None


def _evidence(doc: dict, terms: tuple[str, str]) -> dict:
    sentences = []
    for part in (str(doc.get("title") or ""), str(doc.get("abstract") or "")):
        sentences.extend(sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", part) if sentence.strip())
    for sentence in sentences:
        words = set(tokenize(sentence))
        if all(term in words for term in terms):
            return {"sentence": sentence, "evidence_type": "same_sentence",
                    "word_sentences": {term: sentence for term in terms}}
    selected = {}
    for term in dict.fromkeys(terms):
        selected[term] = next((sentence for sentence in sentences if term in tokenize(sentence)), "")
    distinct = list(dict.fromkeys(sentence for sentence in selected.values() if sentence))
    return {"sentence": "Different source sentences (same document): " + " | ".join(distinct),
            "evidence_type": "separate_sentences", "word_sentences": selected}


def word_pair(index: IRIndex, word1: str, word2: str) -> dict:
    first, second = tokenize(word1), tokenize(word2)
    base = {"word1": first[0] if len(first) == 1 else word1,
            "word2": second[0] if len(second) == 1 else word2}
    if len(first) != 1 or len(second) != 1 or first[0] not in index.vocabulary or second[0] not in index.vocabulary:
        return {**base, "status": "oov", "tfidf_cosine": None, "count_cosine": None,
                "df1": int(index.df[index.vocabulary[first[0]]]) if len(first) == 1 and first[0] in index.vocabulary else 0,
                "df2": int(index.df[index.vocabulary[second[0]]]) if len(second) == 1 and second[0] in index.vocabulary else 0,
                "tf1": int(index.counts[:, index.vocabulary[first[0]]].sum()) if len(first) == 1 and first[0] in index.vocabulary else 0,
                "tf2": int(index.counts[:, index.vocabulary[second[0]]].sum()) if len(second) == 1 and second[0] in index.vocabulary else 0,
                "common_docs": 0, "evidence": []}
    a, b = (index.vocabulary[first[0]], index.vocabulary[second[0]])
    count_a, count_b = index.counts[:, a].toarray().ravel(), index.counts[:, b].toarray().ravel()
    tfidf = index.matrix("tfidf", l2=False)
    tfidf_a, tfidf_b = tfidf[:, a].toarray().ravel(), tfidf[:, b].toarray().ravel()
    common = np.flatnonzero((count_a > 0) & (count_b > 0))
    return {**base, "status": "ok", "tfidf_cosine": cosine(tfidf_a, tfidf_b),
            "count_cosine": cosine(count_a, count_b), "df1": int(index.df[a]),
            "df2": int(index.df[b]), "tf1": int(count_a.sum()), "tf2": int(count_b.sum()),
            "common_docs": int(len(common)),
            "evidence": [{"pmid": str(index.docs[i]["pmid"]), **_evidence(index.docs[i], (first[0], second[0]))} for i in common[:3]]}


def retrieval_metrics(ranked_ids, relevant_ids, k: int = 10) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = {str(x) for x in relevant_ids}
    ranked = [str(x) for x in ranked_ids]
    top = ranked[:k]
    hits = [int(pmid in relevant) for pmid in top]
    count = sum(hits)
    first = next((i + 1 for i, pmid in enumerate(ranked) if pmid in relevant), None)
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    suffix = str(k)
    return {f"precision_at_{suffix}": count / k,
            f"recall_at_{suffix}": count / len(relevant) if relevant else 0.0,
            "mrr": 1 / first if first else 0.0,
            f"ndcg_at_{suffix}": dcg / ideal if ideal else 0.0,
            "no_relevant": not bool(relevant)}
