import json
import math

import numpy as np
import pytest

from ir_core import IRIndex, cosine, load_documents, retrieval_metrics, tokenize, word_pair


def fixture_index():
    return IRIndex([
        {"pmid": "1", "title": "Alpha alpha kidney", "abstract": "Renal alpha."},
        {"pmid": "2", "title": "Beta kidney", "abstract": "Renal beta."},
        {"pmid": "3", "title": "Gamma", "abstract": "Gamma."},
    ])


def test_load_documents_rejects_duplicate_and_empty_text(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('\n'.join([json.dumps({"pmid": "1", "title": "One"}), json.dumps({"pmid": "1", "title": "Two"})]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_documents(path)
    path.write_text(json.dumps({"pmid": "1", "title": "", "abstract": ""}), encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_documents(path)


def test_tokenize_scientific_terms_and_glp1_normalization():
    assert tokenize("GLP-1, glp 1, glucagon-like peptide-1; HbA1c and COVID-19. 2024 5mg") == [
        "glp1", "glp1", "glp1", "hba1c", "and", "covid19", "5mg"
    ]


def test_weight_matrices_and_query_use_same_idf():
    index = fixture_index()
    alpha = index.vocabulary["alpha"]
    gamma = index.vocabulary["gamma"]
    assert index.counts[0, alpha] == 3
    expected = 1 + math.log(4 / 2)
    assert index.idf[alpha] == pytest.approx(expected)
    assert index.matrix("tf")[0, alpha] == 3
    assert index.matrix("log_tf")[0, alpha] == pytest.approx(1 + math.log(3))
    assert index.matrix("idf")[0, alpha] == pytest.approx(expected)
    assert index.matrix("idf")[2, alpha] == 0
    assert index.matrix("tfidf")[0, alpha] == pytest.approx(3 * expected)
    assert index.matrix("log_tfidf")[0, alpha] == pytest.approx((1 + math.log(3)) * expected)
    assert index.query_vector("alpha alpha", "log_tfidf")[0, alpha] == pytest.approx((1 + math.log(2)) * expected)
    assert index.query_vector("unseen", "tf").nnz == 0
    assert index.df[gamma] == 1


def test_rank_excludes_zero_scores_and_stably_breaks_ties():
    index = fixture_index()
    assert index.rank("unseen") == []
    assert [r["pmid"] for r in index.rank("kidney", method="tf", metric="dot")] == ["1", "2"]
    assert index.rank("alpha", method="tf", metric="dot")[0]["score"] == 3
    assert index.rank("alpha", method="tf", metric="cosine")[0]["score"] < 1


def test_cosine_zero_and_word_column_idf_cancellation():
    index = fixture_index()
    assert cosine(np.array([0, 0]), np.array([1, 2])) is None
    assert cosine(np.array([1, 1]), np.array([1, 0])) == pytest.approx(2 ** -0.5)
    result = word_pair(index, "kidney", "renal")
    assert result["status"] == "ok"
    assert result["common_docs"] == 2
    assert result["tf1"] == 2
    assert result["tf2"] == 2
    assert result["tfidf_cosine"] == pytest.approx(result["count_cosine"])
    assert result["tfidf_cosine"] == pytest.approx(1)
    assert word_pair(index, "kidney", "unknown")["status"] == "oov"
    assert word_pair(index, "kidney", "kidney")["tfidf_cosine"] == pytest.approx(1)


def test_retrieval_metrics_hand_computed_full_ranking():
    result = retrieval_metrics(["a", "b", "c", "d"], {"b", "d"}, k=2)
    assert result["precision_at_2"] == pytest.approx(0.5)
    assert result["recall_at_2"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
    assert result["ndcg_at_2"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert retrieval_metrics([], {"a"})["precision_at_10"] == 0
    assert retrieval_metrics(["a"], set())["no_relevant"] is True


def test_silver_labels_include_matching_evidence_without_using_rankings():
    from experiments import build_qrels

    docs = [
        {"pmid": "1", "title": "Semaglutide for weight management", "abstract": ""},
        {"pmid": "2", "title": "Case report", "abstract": "Semaglutide was given. Obesity was discussed."},
    ]
    rows, relevant = build_qrels(docs, {})
    q01 = [r for r in rows if r["qid"] == "q01"]
    assert relevant["q01"] == {"1", "2"}
    assert q01[0]["rule"] == "title_all_groups"
    assert q01[0]["label_source"] == "silver_topic_rule"
    assert q01[1]["rule"] == "abstract_all_groups"
    assert "Semaglutide was given" in q01[1]["evidence"]
    assert "Obesity was discussed" in q01[1]["evidence"]


def test_manual_override_preserves_prior_rule():
    from experiments import build_qrels

    docs = [{"pmid": "1", "title": "Semaglutide weight", "abstract": ""}]
    rows, relevant = build_qrels(docs, {("q01", "1"): (0, "manual_override")})
    row = next(r for r in rows if r["qid"] == "q01")
    assert row["relevant"] == 0
    assert row["rule"] == "title_all_groups"
    assert row["label_source"] == "manual_override"
    assert "override decision: 0" in row["evidence"]
    assert not relevant["q01"]


def test_cbow_sentences_never_cross_title_abstract_or_sentence():
    from experiments import cbow_sentences

    sentences = cbow_sentences([{"pmid": "1", "title": "Alpha beta. Gamma delta.", "abstract": "Epsilon zeta."}])
    assert sentences == [["alpha", "beta"], ["gamma", "delta"], ["epsilon", "zeta"]]


def test_word_pair_evidence_distinguishes_separate_source_sentences():
    index = IRIndex([{"pmid": "1", "title": "Kidney study.", "abstract": "Renal function improved."}])
    evidence = word_pair(index, "kidney", "renal")["evidence"][0]
    assert evidence["evidence_type"] == "separate_sentences"
    assert evidence["sentence"].startswith("Different source sentences (same document):")
    assert "Kidney study." in evidence["sentence"]
    assert "Renal function improved." in evidence["sentence"]


def test_word_pair_evidence_keeps_full_long_same_sentence():
    long_sentence = "Kidney " + ("filler " * 100) + "renal outcomes."
    index = IRIndex([{"pmid": "1", "title": "", "abstract": long_sentence}])
    evidence = word_pair(index, "kidney", "renal")["evidence"][0]
    assert evidence["evidence_type"] == "same_sentence"
    assert evidence["sentence"].endswith("renal outcomes.")
    assert len(evidence["sentence"]) > 400
