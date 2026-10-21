"""Exercise batch orchestration without repeating CBOW training."""
import json
import shutil
import sys
from pathlib import Path
import pandas as pd
import experiments

def test_main_writes_artifacts_and_report(monkeypatch, tmp_path):
    root=Path(experiments.__file__).resolve().parent
    pairs=pd.read_csv(root/'artifacts/word_pairs.csv').to_dict('records')
    cbow=json.loads((root/'artifacts/summary.json').read_text(encoding='utf-8'))['cbow']
    (tmp_path/'data').mkdir()
    corpus=tmp_path/'data/glp1_1000.jsonl'
    shutil.copyfile(root/'data/glp1_1000.jsonl', corpus)
    monkeypatch.setattr(experiments,'ROOT',tmp_path)
    monkeypatch.setattr(experiments,'SOURCE',corpus)
    monkeypatch.setattr(experiments,'ART',tmp_path/'artifacts')
    monkeypatch.setattr(experiments,'cbow_experiment',lambda index:(pairs,cbow))
    monkeypatch.setattr(sys,'argv',['experiments'])
    experiments.main()
    text=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert 'RQ1' in text and 'RQ5' in text and 'Homework 3' in text
    assert (tmp_path/'artifacts/metrics.csv').exists()
    summary=json.loads((tmp_path/'artifacts/summary.json').read_text(encoding='utf-8'))
    assert summary['queries']['count']==12
    assert summary['queries']['judgments']==12000
def test_report_recalculates_interpretation_after_manual_override():
    from ir_core import IRIndex,load_documents
    from report_builder import build_report
    root=Path(experiments.__file__).resolve().parent
    index=IRIndex(load_documents(root/'data/glp1_1000.jsonl'))
    _, relevant=experiments.build_qrels(index.docs,{('q01','39734391'):(0,'manual_review')})
    metrics,macro,_,retrieval=experiments.retrieval_experiment(index,relevant)
    summary=json.loads((root/'artifacts/summary.json').read_text(encoding='utf-8'))
    summary['queries']['override_count']=1
    summary['retrieval']=retrieval
    pairs=pd.read_csv(root/'artifacts/word_pairs.csv').to_dict('records')
    report=build_report(index,macro,metrics,pairs,summary)
    assert '平均約 9.5 篇' in report
    assert '12 題中有 11 題' in report
    assert '12 題的第 1 名都符合規則' not in report
