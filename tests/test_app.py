from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


def launch(section=None):
    assert APP.exists(), 'The requested Streamlit app has not been implemented'
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    if section:
        at.sidebar.radio[0].set_value(section).run()
    assert not at.exception
    return at


def test_overview_shows_identity_and_real_corpus():
    at = launch()
    content = '\n'.join(x.value for x in at.markdown)
    assert 'Homework 3' in content
    assert any(m.value == '1,000' for m in at.metric)


def test_search_known_query_and_oov():
    at = launch('搜尋實驗')
    at.text_input(key='search_query').set_value('semaglutide weight loss')
    at.button(key='search_submit').click().run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 10
    assert 'PMID' in at.dataframe[0].value.columns
    at.text_input(key='search_query').set_value('未知詞 qqqzzznomatch')
    at.button(key='search_submit').click().run()
    assert not at.exception
    assert any('沒有已知詞' in x.value for x in at.warning)
    assert len(at.dataframe) == 0


def test_word_pair_unknown_and_identical_are_safe():
    at = launch('字詞相似度')
    assert len(at.metric) >= 3
    at.text_input(key='word1').set_value('qqqzzznomatch').run()
    assert not at.exception
    assert any(m.value == '未定義' for m in at.metric)
    at.text_input(key='word1').set_value('kidney')
    at.text_input(key='word2').set_value('kidney').run()
    assert not at.exception
    assert any(m.value == '1.0000' for m in at.metric)


def test_microscope_and_evaluation_have_measured_data():
    at = launch('TF-IDF 顯微鏡')
    assert len(at.dataframe) >= 2
    assert not at.exception
    at.sidebar.radio[0].set_value('RQ1–RQ5 / 評估').run()
    assert not at.exception
    assert any('nDCG@10' in df.value.columns for df in at.dataframe)


def test_example_query_updates_editable_search():
    at = launch('搜尋實驗')
    at.selectbox(key='search_example').set_value(at.selectbox(key='search_example').options[1]).run()
    assert at.text_input(key='search_query').value == 'liraglutide obesity'


def test_missing_artifacts_gives_rebuild_instruction(tmp_path):
    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()
    source = APP.read_text(encoding='utf-8')
    source = source.replace('ROOT = Path(__file__).resolve().parent', f'ROOT = Path({str(tmp_path)!r})')
    at = AppTest.from_string(source, default_timeout=30).run()
    assert not at.exception
    assert len(at.error) == 1
    assert any('python -m experiments' in c.value for c in at.code)


def test_negative_cbow_similarity_is_preserved():
    at = launch('字詞相似度')
    at.text_input(key='word1').set_value('agonist')
    at.text_input(key='word2').set_value('receptor').run()
    assert not at.exception
    assert float(at.metric[1].value) < 0
    assert float(at.metric[2].value) > 0
