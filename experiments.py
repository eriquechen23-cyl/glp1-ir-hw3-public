"""Offline, reproducible GLP-1 weighting and word-vector experiments.

Run: .venv/Scripts/python.exe experiments.py [--overrides path/to/labels.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ir_core import IRIndex, cosine, load_documents, retrieval_metrics, tokenize, word_pair


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SOURCE = ROOT / "data" / "glp1_1000.jsonl"

# Fixed before running retrieval. These are broad topic rules, not clinical judgments.
QUERIES = [
    ("q01", "semaglutide weight loss", "Semaglutide and weight reduction", [["semaglutide"], ["weight", "obesity", "overweight"]]),
    ("q02", "liraglutide obesity", "Liraglutide and obesity", [["liraglutide"], ["obesity", "weight", "overweight"]]),
    ("q03", "tirzepatide diabetes", "Tirzepatide in diabetes", [["tirzepatide"], ["diabetes", "glycemic", "glucose"]]),
    ("q04", "cardiovascular outcomes", "Cardiovascular outcomes", [["cardiovascular", "cardiac", "heart"], ["outcomes", "outcome", "events", "event", "risk"]]),
    ("q05", "kidney disease renal outcomes", "Kidney disease and renal outcomes", [["kidney", "renal"], ["disease", "outcomes", "outcome", "function", "injury"]]),
    ("q06", "glp1 pancreatitis", "Pancreatitis and GLP-1", [["pancreatitis"], ["glp1", "agonist", "agonists"]]),
    ("q07", "gastric emptying glp1", "Gastric emptying", [["gastric"], ["emptying"]]),
    ("q08", "nausea vomiting adverse effects", "Nausea or vomiting as adverse effects", [["nausea", "vomiting"], ["adverse", "side", "effects", "events", "safety"]]),
    ("q09", "insulin secretion glp1", "Insulin secretion", [["insulin"], ["secretion", "secretory"]]),
    ("q10", "appetite food intake", "Appetite and food intake", [["appetite", "food"], ["intake", "feeding"]]),
    ("q11", "liver steatosis glp1", "Liver steatosis", [["liver", "hepatic"], ["steatosis", "fatty", "masld"]]),
    ("q12", "alzheimer neurodegeneration glp1", "Alzheimer disease and neurodegeneration", [["alzheimer", "neurodegeneration", "neurodegenerative"], ["glp1", "agonist", "agonists"]]),
]

PAIRS = [
    ("semaglutide", "liraglutide", "related drugs"), ("obesity", "overweight", "related condition terms"),
    ("nausea", "vomiting", "co-reported symptoms"), ("insulin", "glucose", "metabolic association"),
    ("kidney", "renal", "possible lexical alternatives"), ("cardiovascular", "cardiac", "related organ terms"),
    ("appetite", "satiety", "related feeding terms"), ("diabetes", "obesity", "co-studied conditions"),
    ("agonist", "receptor", "mechanistic association"), ("weight", "loss", "common phrase"),
    ("placebo", "randomized", "trial association"), ("pancreas", "pancreatitis", "organ and condition"),
]

METHODS = [(name, name, "cosine") for name in ("tf", "log_tf", "idf", "tfidf", "log_tfidf")]
METHODS += [("tfidf_dot", "tfidf", "dot"), ("log_tfidf_dot", "log_tfidf", "dot")]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _matches(text: str, groups: list[list[str]]) -> bool:
    terms = set(tokenize(text))
    return all(bool(terms.intersection(group)) for group in groups)


def _snippet(text: str, groups: list[list[str]]) -> str:
    text_terms = set(tokenize(text))
    hits = [sorted(text_terms.intersection(group)) for group in groups]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chosen = []
    for group_hit in hits:
        for sentence in sentences:
            if set(tokenize(sentence)).intersection(group_hit):
                if sentence not in chosen:
                    chosen.append(sentence)
                break
    return "matched terms by group: " + json.dumps(hits, ensure_ascii=False) + "; source sentences: " + " | ".join(chosen)


def load_overrides(path: Path | None) -> dict[tuple[str, str], tuple[int, str]]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            relevant = row["relevant"].strip()
            if relevant not in {"0", "1"}:
                raise ValueError(f"override relevant must be 0 or 1: {row}")
            result[(row["qid"].strip(), row["pmid"].strip())] = (int(relevant), row.get("label_source", "manual_override") or "manual_override")
    return result


def build_qrels(docs: list[dict], overrides: dict) -> tuple[list[dict], dict[str, set[str]]]:
    known_qids = {q[0] for q in QUERIES}
    known_pmids = {str(d["pmid"]) for d in docs}
    unknown = set(overrides) - {(q, p) for q in known_qids for p in known_pmids}
    if unknown:
        raise ValueError(f"unknown override IDs: {sorted(unknown)[:3]}")
    rows = []
    relevant = {qid: set() for qid in known_qids}
    for qid, _query, _intent, groups in QUERIES:
        for doc in docs:
            pmid = str(doc["pmid"])
            title, abstract = str(doc.get("title") or ""), str(doc.get("abstract") or "")
            if _matches(title, groups):
                label, rule, evidence = 1, "title_all_groups", _snippet(title, groups)
            elif _matches(abstract, groups):
                label, rule, evidence = 1, "abstract_all_groups", _snippet(abstract, groups)
            else:
                title_terms, abstract_terms = set(tokenize(title)), set(tokenize(abstract))
                title_missing = [group for group in groups if not title_terms.intersection(group)]
                abstract_missing = [group for group in groups if not abstract_terms.intersection(group)]
                label, rule = 0, "no_topic_rule_match"
                evidence = (f"PMID {pmid}; no all-groups match; title missing groups: {title_missing}; "
                            f"abstract missing groups: {abstract_missing}")
            label_source = "silver_topic_rule"
            if (qid, pmid) in overrides:
                original_evidence = evidence
                label, label_source = overrides[(qid, pmid)]
                evidence = (f"override decision: {label}; original rule: {rule}; "
                            f"original evidence: {original_evidence}; source PMID {pmid}")
            rows.append({"qid": qid, "pmid": pmid, "relevant": label, "rule": rule,
                         "label_source": label_source, "evidence": evidence})
            if label:
                relevant[qid].add(pmid)
    return rows, relevant


def retrieval_experiment(index: IRIndex, relevant: dict[str, set[str]]) -> tuple[list[dict], list[dict], list[dict], dict]:
    metric_rows, top_rows = [], []
    first_example = {}
    max_norm_error = 0.0
    l2_rank_mismatches = 0
    for qid, query, _intent, _groups in QUERIES:
        baseline = index.rank(query, "log_tfidf", "cosine", limit=None)
        baseline_ranks = {r["pmid"]: i + 1 for i, r in enumerate(baseline)}
        for label, method, metric in METHODS:
            ranked = index.rank(query, method, metric, limit=None)
            values = retrieval_metrics([r["pmid"] for r in ranked], relevant[qid])
            metric_rows.append({"qid": qid, "query": query, "method": label, "relevant_count": len(relevant[qid]),
                                "retrieved_count": len(ranked), **values})
            for rank, hit in enumerate(ranked[:10], 1):
                baseline_rank = baseline_ranks.get(hit["pmid"])
                top_rows.append({"qid": qid, "query": query, "method": label, "rank": rank,
                                 "pmid": hit["pmid"], "title": hit["title"], "score": hit["score"],
                                 "silver_relevant": int(hit["pmid"] in relevant[qid]),
                                 "baseline_log_tfidf_cosine_rank": baseline_rank or "",
                                 "rank_change_vs_baseline": baseline_rank - rank if baseline_rank else "outside_positive_baseline"})
            if label == "log_tfidf" and qid == "q01":
                first_example = {"qid": qid, "top_pmid": ranked[0]["pmid"] if ranked else None,
                                 "top_title": ranked[0]["title"] if ranked else None}
        for method in ("tfidf", "log_tfidf"):
            raw_q = index.query_vector(query, method)
            raw_docs = index.matrix(method)
            raw_dot = np.asarray(raw_docs.dot(raw_q.T).toarray()).ravel()
            doc_norm = np.sqrt(np.asarray(raw_docs.multiply(raw_docs).sum(axis=1)).ravel())
            q_norm = math.sqrt(float(raw_q.multiply(raw_q).sum()))
            direct_cosine = np.divide(raw_dot, doc_norm * q_norm,
                                      out=np.zeros_like(raw_dot), where=(doc_norm * q_norm) > 0)
            l2_q = index.query_vector(query, method, l2=True)
            l2_scores = np.asarray(index.matrix(method, l2=True).dot(l2_q.T).toarray()).ravel()
            max_norm_error = max(max_norm_error, float(np.max(np.abs(direct_cosine - l2_scores))))
            ranked_indices = [r["doc_index"] for r in index.rank(query, method, "cosine", limit=None)]
            expected_order = sorted(np.flatnonzero(l2_scores > 0), key=lambda i: (-l2_scores[i], int(i)))
            l2_rank_mismatches += int(ranked_indices != expected_order)
    macro = []
    for label, _method, _metric in METHODS:
        sub = [row for row in metric_rows if row["method"] == label]
        macro.append({"method": label, "queries": len(sub), "mean_relevant_count": float(np.mean([r["relevant_count"] for r in sub])),
                      **{key: float(np.mean([r[key] for r in sub])) for key in ("precision_at_10", "recall_at_10", "mrr", "ndcg_at_10")},
                      **{key + "_sd": float(np.std([r[key] for r in sub], ddof=1)) for key in ("precision_at_10", "recall_at_10", "mrr", "ndcg_at_10")}})
    return metric_rows, macro, top_rows, {"cosine_prior_l2_max_abs_error": max_norm_error,
                                         "l2_rank_mismatch_count": l2_rank_mismatches, "example": first_example}


def term_analysis(index: IRIndex) -> tuple[list[dict], list[dict]]:
    n = len(index.docs)
    terms = ["glp1", "agonist", "receptor", "semaglutide", "liraglutide", "tirzepatide", "obesity", "diabetes", "kidney", "renal", "pancreatitis", "satiety", "alzheimer"]
    by_df = np.argsort(-index.df, kind="stable")[:20]
    rare = np.flatnonzero(index.df == 1)[:20]
    terms.extend([str(index.terms[i]) for i in by_df])
    terms.extend([str(index.terms[i]) for i in rare])
    rows = []
    for term in dict.fromkeys(terms):
        if term in index.vocabulary:
            j = index.vocabulary[term]
            rows.append({"term": term, "df": int(index.df[j]), "df_fraction": float(index.df[j] / n),
                         "idf": float(index.idf[j]), "collection_tf": int(index.counts[:, j].sum())})
        else:
            rows.append({"term": term, "df": 0, "df_fraction": 0.0, "idf": "", "collection_tf": 0})
    demo = []
    for term in ("glp1", "semaglutide", "obesity", "diabetes", "agonist"):
        if term not in index.vocabulary:
            continue
        j = index.vocabulary[term]
        counts = index.counts[:, j].toarray().ravel()
        for i in np.argsort(-counts, kind="stable")[:5]:
            if counts[i] == 0:
                continue
            demo.append({"term": term, "pmid": str(index.docs[i]["pmid"]), "title": index.docs[i].get("title", ""),
                         "tf": int(counts[i]), "log_tf": float(1 + math.log(counts[i])), "idf": float(index.idf[j]),
                         "tfidf": float(counts[i] * index.idf[j]),
                         "log_tfidf": float((1 + math.log(counts[i])) * index.idf[j])})
    return rows, demo


def stable_hash(word: str) -> int:
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")


def cbow_sentences(docs: list[dict]) -> list[list[str]]:
    sentences = []
    for doc in docs:
        for part in (str(doc.get("title") or ""), str(doc.get("abstract") or "")):
            for sentence in re.split(r"(?<=[.!?])\s+", part):
                tokens = tokenize(sentence)
                if tokens:
                    sentences.append(tokens)
    return sentences


def cbow_experiment(index: IRIndex) -> tuple[list[dict], dict]:
    import gensim
    from gensim.models import Word2Vec
    from scipy.stats import spearmanr

    sentences = cbow_sentences(index.docs)
    configs = {"architecture": "CBOW", "sg": 0, "vector_size": 100, "window": 5, "min_count": 3,
               "epochs": 50, "workers": 1, "negative": 10, "sample": 1e-3,
               "seeds": [17, 42, 73], "hashfxn": "sha256_first8_little_endian",
               "sentence_count": len(sentences), "gensim_version": gensim.__version__}
    seed_models = {}
    for seed in configs["seeds"]:
        seed_models[seed] = Word2Vec(sentences=sentences, sg=0, vector_size=100, window=5,
                                     min_count=3, epochs=50, workers=1, negative=10,
                                     sample=1e-3, seed=seed, hashfxn=stable_hash)
    model = seed_models[42]
    words = np.asarray(model.wv.index_to_key, dtype=str)
    vectors = np.asarray(model.wv.vectors, dtype=np.float32)
    centered = vectors.astype(np.float64) - vectors.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coords = (centered @ vt[:2].T).astype(np.float32)
    ratios = (singular[:2] ** 2 / np.sum(singular ** 2)).astype(np.float64)
    np.savez_compressed(ART / "cbow_vectors.npz", words=words, vectors=vectors, coords=coords,
                        explained_variance_ratio=ratios)
    word_to_i = {word: i for i, word in enumerate(words)}
    rows = []
    original_for_corr, projected_for_corr = [], []
    max_idf_delta = 0.0
    for word1, word2, hypothesis in PAIRS:
        lexical = word_pair(index, word1, word2)
        if lexical["status"] == "ok":
            max_idf_delta = max(max_idf_delta, abs(lexical["tfidf_cosine"] - lexical["count_cosine"]))
        seed_scores = {seed: (float(m.wv.similarity(word1, word2)) if word1 in m.wv and word2 in m.wv else None)
                       for seed, m in seed_models.items()}
        valid = [v for v in seed_scores.values() if v is not None]
        original = cosine(vectors[word_to_i[word1]], vectors[word_to_i[word2]]) if word1 in word_to_i and word2 in word_to_i else None
        projected = cosine(coords[word_to_i[word1]], coords[word_to_i[word2]]) if word1 in word_to_i and word2 in word_to_i else None
        if original is not None and projected is not None:
            original_for_corr.append(original)
            projected_for_corr.append(projected)
        neighbors1 = model.wv.most_similar(word1, topn=5) if word1 in model.wv else []
        neighbors2 = model.wv.most_similar(word2, topn=5) if word2 in model.wv else []
        rows.append({"word1": word1, "word2": word2, "relationship_hypothesis": hypothesis,
                     "status": lexical["status"], "df1": lexical["df1"], "df2": lexical["df2"],
                     "tf1": lexical["tf1"], "tf2": lexical["tf2"],
                     "common_docs": lexical["common_docs"], "tfidf_column_cosine": lexical["tfidf_cosine"],
                     "count_column_cosine": lexical["count_cosine"], "cbow_100d_seed42_cosine": original,
                     "cbow_pca2d_seed42_cosine": projected,
                     "pca_abs_delta": abs(original - projected) if original is not None and projected is not None else None,
                     "cbow_seed17_cosine": seed_scores[17], "cbow_seed42_cosine": seed_scores[42],
                     "cbow_seed73_cosine": seed_scores[73], "cbow_mean": float(np.mean(valid)) if valid else None,
                     "cbow_sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                     "evidence": json.dumps(lexical["evidence"], ensure_ascii=False),
                     "neighbors1": json.dumps(neighbors1, ensure_ascii=False),
                     "neighbors2": json.dumps(neighbors2, ensure_ascii=False)})
    rho = float(spearmanr(original_for_corr, projected_for_corr).statistic) if len(original_for_corr) >= 3 else None
    return rows, {"config": configs, "vocabulary_size": len(words),
                  "explained_variance_ratio": ratios.tolist(), "explained_variance_sum": float(ratios.sum()),
                  "pair_rank_spearman_100d_vs_pca2d": rho,
                  "max_idf_column_cosine_delta": max_idf_delta,
                  "max_pair_pca_abs_delta": max((r["pca_abs_delta"] or 0 for r in rows), default=0)}


def write_report(index, macro, metrics, pairs, summary) -> None:
    from report_builder import build_report
    (ROOT / "report.md").write_text(build_report(index, macro, metrics, pairs, summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=Path, help="Optional qid,pmid,relevant,label_source CSV")
    args = parser.parse_args()
    ART.mkdir(exist_ok=True)
    docs = load_documents(SOURCE)
    index = IRIndex(docs)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    queries = [{"qid": qid, "query": query, "intent": intent,
                "title_rule": {"all_groups": groups}, "abstract_rule": {"all_groups": groups}}
               for qid, query, intent, groups in QUERIES]
    write_json(ART / "queries.json", queries)
    qrels, relevant = build_qrels(docs, load_overrides(args.overrides))
    write_csv(ART / "qrels.csv", qrels)
    metrics, macro, top, retrieval_summary = retrieval_experiment(index, relevant)
    write_csv(ART / "metrics.csv", metrics)
    write_csv(ART / "metrics_macro.csv", macro)
    write_csv(ART / "top10.csv", top)
    stats, demo = term_analysis(index)
    write_csv(ART / "term_statistics.csv", stats)
    write_csv(ART / "tf_demo.csv", demo)
    pairs, cbow = cbow_experiment(index)
    write_csv(ART / "word_pairs.csv", pairs)
    summary = {"corpus": {"documents": len(docs), "vocabulary_size": len(index.terms),
                          "token_count": sum(map(len, index.tokens)), "sha256": digest,
                          "source": str(SOURCE.relative_to(ROOT)), "source_manifest": "data/glp1_1000.manifest.json"},
               "queries": {"count": len(queries), "silver_positive_counts": {q: len(v) for q, v in relevant.items()},
                           "judgments": len(qrels), "override_count": len(load_overrides(args.overrides))},
               "retrieval": retrieval_summary, "cbow": cbow,
               "conclusions": {"top_ndcg_method_under_silver_labels": max(macro, key=lambda r: r["ndcg_at_10"])["method"],
                               "idf_cancels_in_unnormalized_word_column_cosine": True,
                               "pca_2d_is_diagnostic_only": True},
               "software": {"python": platform.python_version(), "numpy": np.__version__},
               "term_statistics": stats, "tf_demo": demo}
    write_json(ART / "summary.json", summary)
    write_report(index, macro, metrics, pairs, summary)
    print(json.dumps({"documents": len(docs), "terms": len(index.terms), "queries": len(queries),
                      "qrels": len(qrels), "cbow_terms": cbow["vocabulary_size"],
                      "pca_variance": cbow["explained_variance_sum"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
