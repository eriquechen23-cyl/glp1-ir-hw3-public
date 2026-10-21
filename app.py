"""Offline GLP-1 weighting and word embedding research demo."""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ir_core import IRIndex, load_documents, tokenize, word_pair, cosine

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
METHODS = ['tf', 'log_tf', 'idf', 'tfidf', 'log_tfidf']
SECTIONS = ['概覽', '搜尋實驗', 'TF-IDF 顯微鏡', '字詞相似度', 'RQ1–RQ5 / 評估']
METRIC_NAMES = {'precision_at_10': 'P@10', 'recall_at_10': 'Recall@10', 'mrr': 'MRR', 'ndcg_at_10': 'nDCG@10'}
st.set_page_config(page_title='GLP-1 Term Weighting Lab', page_icon='🔬', layout='wide')


@st.cache_resource
def load_index():
    return IRIndex(load_documents(ROOT / 'data/glp1_1000.jsonl'))


@st.cache_resource
def load_embedding():
    with np.load(ART / 'cbow_vectors.npz', allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


@st.cache_data
def load_artifacts(version=None):
    tables = {p.stem: pd.read_csv(p, dtype={'pmid': str}) for p in ART.glob('*.csv')}
    return (json.loads((ART / 'summary.json').read_text(encoding='utf-8')),
            json.loads((ART / 'queries.json').read_text(encoding='utf-8')), tables,
            (ROOT / 'report.md').read_text(encoding='utf-8'))


def download(path, label=None):
    if path.exists():
        st.download_button(label or f'下載 {path.name}', path.read_bytes(), file_name=path.name,
                           key=f'download_{path.name}', mime='text/csv' if path.suffix == '.csv' else None)


def show_results(hits, query):
    if not hits:
        st.info('沒有正分結果。')
        return
    rows = []
    for rank, hit in enumerate(hits, 1):
        doc = index.docs[hit['doc_index']]
        matches = sorted(set(tokenize(query)) & set(index.tokens[hit['doc_index']]))
        rows.append({'名次': rank, 'PMID': hit['pmid'], '標題': hit['title'], 'score': hit['score'],
                     '命中詞': ', '.join(matches), 'PubMed': f"https://pubmed.ncbi.nlm.nih.gov/{hit['pmid']}/"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                 column_config={'PubMed': st.column_config.LinkColumn('PubMed', display_text='原文'),
                                'score': st.column_config.NumberColumn(format='%.6f')})
    for hit in hits:
        with st.expander(f"{hit['pmid']} · {hit['title']}"):
            st.write(index.docs[hit['doc_index']].get('abstract') or '此紀錄沒有摘要。')


def overview():
    st.subheader('同一批文章，換一種計分方式，搜尋結果會變嗎？')
    st.write('做了什麼：用 1,000 篇文章和 12 個固定問題，比較五種詞權重的搜尋結果。每篇只使用標題和摘要。')
    cols = st.columns(4)
    values = [('文章', summary['corpus']['documents']), ('不同詞', summary['corpus']['vocabulary_size']),
              ('總詞次', summary['corpus']['token_count']), ('固定查詢', summary['queries']['count'])]
    for col, (label, value) in zip(cols, values):
        col.metric(label, f'{value:,}')
    a, b = st.columns(2)
    with a:
        st.markdown('**怎麼計分**')
        st.write('TF 計算一個詞在文章出現幾次；log TF 降低重複次數的影響。IDF 給較少文章出現的詞較高權重。TF-IDF 把 TF 和 IDF 相乘。')
        st.markdown('**結果**')
        result_by_method = tables['metrics_macro'].set_index('method')
        st.write(f"同一個字出現很多次，不代表文章更符合搜尋問題。本次前10篇符合規則的平均篇數：TF {result_by_method.loc['tf','precision_at_10']*10:.1f} 篇，Log TF {result_by_method.loc['log_tf','precision_at_10']*10:.1f} 篇。")
    with b:
        st.markdown('**另一個比較：兩個詞的用法接近嗎？**')
        st.write('CBOW 從詞附近的用字，學習每個詞的數字表示（word embedding）。「字詞相似度」頁會比較兩個詞，也列出原文句子。2D 圖會丟失資訊，應以原始 100D 結果為主。')
        st.info('我的看法：依固定關鍵字規則列出的參考文章（silver labels），只能當本次比較依據，還沒人工逐篇確認。搜尋分數高，不代表文章真的回答問題，也不能證明療效。')
    measured = tables['metrics_macro'].rename(columns=METRIC_NAMES)
    st.dataframe(measured[['method', 'P@10', 'Recall@10', 'MRR', 'nDCG@10']].round(3), hide_index=True, width='stretch')
    st.caption('前五種方法使用 cosine similarity；名稱含 _dot 的兩種是未做 L2 normalization 的 dot product 對照。')
    if summary['queries'].get('override_count', 0):
        st.info('目前已套用人工修訂名單，指標與報告已依修訂名單更新。')
    download(ROOT / 'report.md', '下載完整研究報告')
    download(ART / 'metrics_macro.csv')
    with st.expander('資料處理、來源與可重現性'):
        st.write('英文小寫，GLP-1 寫法統一為 glp1；保留科學詞，不合併同義詞、不做 stemming，也不移除 stopwords，讓各權重的影響可直接比較。單獨數字不保留。')
        st.code(f"SHA-256: {summary['corpus']['sha256']}")
        st.json(summary['cbow']['config'])
        st.markdown('[PubMed](https://pubmed.ncbi.nlm.nih.gov/) · [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) · [PCA 文件](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)')


def search():
    st.subheader('搜尋實驗｜只改一個條件，觀察前十名')
    def choose_query():
        chosen = st.session_state['search_example']
        st.session_state['search_query'] = chosen['query'] if isinstance(chosen, dict) else next(q['query'] for q in queries if f"{q['qid']} · {q['query']}" == chosen)
    example = st.selectbox('固定查詢範例', queries, format_func=lambda q: f"{q['qid']} · {q['query']}", key='search_example', on_change=choose_query)
    with st.form('search_form'):
        query = st.text_input('輸入英文查詢', value=example['query'], key='search_query')
        left, right = st.columns(2)
        method = left.selectbox('詞權重', METHODS, index=4)
        metric = right.selectbox('比較方式', ['cosine', 'dot'], format_func=lambda m: 'cosine similarity（預設）' if m == 'cosine' else 'dot product（未做 L2 normalization 對照）')
        submitted = st.form_submit_button('開始搜尋', key='search_submit', type='primary')
    st.caption('選範例後會自動填入，也能自行修改。cosine similarity 會除以向量長度；dot product 不會，因此容易受重複用字影響。')
    if submitted:
        st.session_state['search_result'] = (query, method, metric)
    if 'search_result' not in st.session_state:
        return
    query, method, metric = st.session_state['search_result']
    known = set(tokenize(query)) & set(index.vocabulary)
    if not known:
        st.warning('沒有已知詞：請輸入語料中的英文詞，例如 semaglutide、kidney；中文或未知詞不會產生任意結果。')
        return
    unknown = set(tokenize(query)) - set(index.vocabulary)
    if unknown:
        st.info('忽略 OOV 詞：' + ', '.join(sorted(unknown)))
    st.caption(f'本次：{query} · {method} · {metric}')
    hits = index.rank(query, method, metric)
    show_results(hits, query)
    base = index.rank(query, 'log_tfidf', 'cosine', limit=None)
    base_ranks = {h['pmid']: r for r, h in enumerate(base, 1)}
    st.markdown('**和 log TF-IDF cosine similarity 基準比較**')
    st.dataframe(pd.DataFrame([{'PMID': h['pmid'], '目前名次': r, '基準名次': base_ranks.get(h['pmid']),
                                 '名次改善': base_ranks.get(h['pmid'], r) - r} for r, h in enumerate(hits, 1)]), hide_index=True)


def microscope():
    st.subheader('TF-IDF 顯微鏡｜把一篇文章的分數拆開')
    d = st.selectbox('搜尋 PMID 或標題', range(len(index.docs)),
                     format_func=lambda i: f"{index.docs[i]['pmid']} · {index.docs[i]['title']}")
    st.markdown(f"[PubMed PMID {index.docs[d]['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{index.docs[d]['pmid']}/)")
    terms = sorted(set(index.tokens[d]))
    term = st.selectbox('選一個文章中的詞', terms, index=terms.index('glp1') if 'glp1' in terms else 0)
    j = index.vocabulary[term]
    tf = int(index.counts[d, j])
    logtf = 1 + np.log(tf)
    vals = [('TF', tf), ('log TF', logtf), ('DF', int(index.df[j])), ('IDF', index.idf[j]),
            ('TF-IDF', tf * index.idf[j]), ('log TF-IDF', logtf * index.idf[j])]
    for col, (label, val) in zip(st.columns(6), vals):
        col.metric(label, str(val) if isinstance(val, int) else f'{val:.4f}')
    st.latex(r'\mathrm{logTF}=1+\ln(tf)\;(tf>0),\quad \mathrm{IDF}=\ln\frac{N+1}{df+1}+1')
    st.latex(r'\hat w_{d,t}=\frac{w_{d,t}}{\sqrt{\sum_j w_{d,j}^{2}}},\quad \cos(d,q)=\sum_t \hat w_{d,t}\hat w_{q,t}')
    query = st.text_input('拆解查詢詞的得分貢獻', 'semaglutide weight loss', key='micro_query')
    raw = index.matrix('log_tfidf')[d]
    norm = float(np.sqrt(raw.multiply(raw).sum()))
    qraw = index.query_vector(query, 'log_tfidf')
    qnorm = float(np.sqrt(qraw.multiply(qraw).sum()))
    nd = index.matrix('log_tfidf', l2=True)[d]
    nq = index.query_vector(query, 'log_tfidf', l2=True)
    rows = []
    for t in dict.fromkeys(tokenize(query)):
        v = index.vocabulary.get(t)
        rows.append({'詞': t, 'TF': int(index.counts[d, v]) if v is not None else 0,
                     'DF': int(index.df[v]) if v is not None else None, 'IDF': float(index.idf[v]) if v is not None else None,
                     '文件 raw weight': float(raw[0, v]) if v is not None else 0,
                     '查詢 raw weight': float(qraw[0, v]) if v is not None else 0,
                     'cosine 貢獻': float(nd[0, v] * nq[0, v]) if v is not None else 0})
    st.caption(f'log TF-IDF 文件 L2 分母 = {norm:.6f}；查詢 L2 分母 = {qnorm:.6f}；下表貢獻相加就是這篇的 cosine similarity。OOV 詞權重為 0。')
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    a, b = st.tabs(['RQ1 · 重複次數的影響', 'RQ2 · 常見詞與罕見詞'])
    with a:
        x = np.arange(1, 21)
        fig = go.Figure([go.Scatter(x=x, y=x, name='TF'), go.Scatter(x=x, y=1 + np.log(x), name='log TF')])
        fig.update_layout(xaxis_title='詞在單篇出現次數', yaxis_title='權重', height=320)
        st.plotly_chart(fig, width='stretch')
        st.dataframe(tables['tf_demo'], hide_index=True, width='stretch')
    with b:
        st.write('DF 看「幾篇有這個詞」，不是總共出現幾次；IDF 讓少見詞有較大權重。')
        stats = tables['term_statistics']
        st.dataframe(stats[stats.term.isin(['glp1', 'of', 'and', 'pancreatitis', 'semaglutide', 'alzheimer'])].sort_values('idf', ascending=False), hide_index=True)
        st.dataframe(stats.sort_values(['df', 'term']), hide_index=True, width='stretch')


def neighbors(word, vectors, vocabulary, n=5):
    if word not in vocabulary:
        return []
    v = vocabulary[word]
    base = vectors[v]
    lengths = np.linalg.norm(vectors, axis=1)
    denom = lengths * np.linalg.norm(base)
    scores = np.divide(vectors @ base, denom, out=np.full(len(vectors), -np.inf), where=denom > 0)
    scores[v] = -np.inf
    order = np.argsort(-scores, kind='stable')[:n]
    words = list(vocabulary)
    return [(words[i], float(scores[i])) for i in order if np.isfinite(scores[i])]


@st.cache_resource
def word_columns():
    # Columns stay unnormalized by document: only column lengths cancel in cosine.
    return index.matrix('tfidf', l2=False).T.toarray()


def word_similarity():
    st.subheader('WORD1 / WORD2｜同一對詞，三種分數怎麼比？')
    pairs = tables['word_pairs']
    choices = [f'{r.word1} / {r.word2}' for r in pairs.itertuples()]
    default = choices.index('kidney / renal') if 'kidney / renal' in choices else 0
    def choose_pair():
        st.session_state['word1'], st.session_state['word2'] = st.session_state['pair_choice'].split(' / ')
    selected = st.selectbox('預設詞對', choices, index=default, key='pair_choice', on_change=choose_pair)
    first, second = selected.split(' / ')
    a, b = st.columns(2)
    w1 = a.text_input('WORD1（單一英文詞）', value=first, key='word1')
    w2 = b.text_input('WORD2（單一英文詞）', value=second, key='word2')
    pair = word_pair(index, w1, w2)
    w1, w2 = pair['word1'], pair['word2']
    evocab = {str(w): i for i, w in enumerate(embedding['words'])}
    cbow = cosine(embedding['vectors'][evocab[w1]], embedding['vectors'][evocab[w2]]) if w1 in evocab and w2 in evocab else None
    pca = cosine(embedding['coords'][evocab[w1]], embedding['coords'][evocab[w2]]) if w1 in evocab and w2 in evocab else None
    scores = [pair['tfidf_cosine'], cbow, pca]
    for col, label, value in zip(st.columns(3), ['TF-IDF：文章中的出現分布', 'CBOW：原始 100D', 'PCA：縮成 2D'], scores):
        col.metric(label, '未定義' if value is None else f'{value:.4f}')
    if any(s is None for s in scores):
        st.warning('OOV、輸入非單一詞或向量長度為 0：相應的 cosine similarity 未定義。CBOW min_count=3，詞表比 TF-IDF 小。')
    st.caption(f"{w1}: TF={pair['tf1']}, DF={pair['df1']} · {w2}: TF={pair['tf2']}, DF={pair['df2']} · 共同文章={pair['common_docs']}")
    threshold = st.slider('探索用 similarity threshold', -1.0, 1.0, 0.5, 0.05)
    passed = lambda x: '未定義' if x is None else ('是' if x >= threshold else '否')
    agree = '未定義' if scores[0] is None or cbow is None else ('一致' if (scores[0] >= threshold) == (cbow >= threshold) else '不一致')
    st.dataframe(pd.DataFrame([{'WORD1': w1, 'WORD2': w2, 'TF-IDF': scores[0], 'TF-IDF 達門檻': passed(scores[0]),
                                'CBOW 100D': cbow, 'CBOW 達門檻': passed(cbow), 'PCA 2D': pca, 'agreement': agree}]), hide_index=True)
    st.info('這個門檻由你設定，只方便比較。兩種方法都超過門檻，不代表兩個詞已證明同義。主要看 CBOW 100D；PCA 2D 只用來看位置。分數可以是負值。')
    st.write('TF-IDF 看兩個字是否常出現在相同的文章；CBOW 看它們附近常用哪些字。PCA 2D 方便畫圖，但可能改變原本的分數。')
    with st.expander('詞對分數怎麼計算？（技術細節）'):
        st.write('每個字在 1,000 篇文章上各有一個分數，組成 word-column vector，再計算 cosine similarity。這裡不先做文件 L2 normalization；固定一個字的 IDF 倍數在計算時會抵消，所以 TF-IDF 與 TF 的詞對分數相同。這些分數不能直接證明同義。')
    evidence_tab, neighbors_tab, plot_tab = st.tabs(['原文證據與穩定性', '最近五個詞', 'PCA 固定投影'])
    with evidence_tab:
        for item in pair['evidence']:
            st.markdown(f"[PMID {item['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{item['pmid']}/) · {item['evidence_type']}")
            st.write(item['sentence'])
        if not pair['evidence']:
            st.caption('沒有共同文章證據；以下各列是兩個詞各自的出現位置，不代表同句或共篇。')
            for word in dict.fromkeys([w1, w2]):
                if word in index.vocabulary:
                    ids = index.counts[:, index.vocabulary[word]].nonzero()[0][:2]
                    for i in ids:
                        doc = index.docs[i]
                        sentence = next((s for s in re.split(r'(?<=[.!?])\s+', str(doc.get('title', '')) + ' ' + str(doc.get('abstract', ''))) if word in tokenize(s)), '')
                        st.markdown(f"**{word}** · [PMID {doc['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{doc['pmid']}/)")
                        st.write(sentence)
        preset = pairs[((pairs.word1 == w1) & (pairs.word2 == w2)) | ((pairs.word1 == w2) & (pairs.word2 == w1))]
        if len(preset):
            st.dataframe(preset[['cbow_seed17_cosine', 'cbow_seed42_cosine', 'cbow_seed73_cosine', 'cbow_mean', 'cbow_sd']], hide_index=True)
            st.caption('同一資料訓練三次的 CBOW 100D 結果。mean 是平均；SD 越大，表示三次結果差異越大。')
        else:
            st.caption('此自訂詞對沒有三個 seeds 的預先計算結果；不推估平均與 SD。')
    with neighbors_tab:
        for word in dict.fromkeys([w1, w2]):
            left, right = st.columns(2)
            left.markdown(f'**{word} · TF-IDF columns**')
            left.dataframe(pd.DataFrame(neighbors(word, word_columns(), index.vocabulary), columns=['word', 'cosine similarity']), hide_index=True)
            right.markdown(f'**{word} · CBOW 100D**')
            right.dataframe(pd.DataFrame(neighbors(word, embedding['vectors'], evocab), columns=['word', 'cosine similarity']), hide_index=True)
    with plot_tab:
        variance = embedding['explained_variance_ratio']
        st.warning(f'PC1={variance[0]:.2%}；PC2={variance[1]:.2%}；合計保留 {sum(variance):.2%} 變異。降維會改變角度，不能以平面距離驗證原始相似度。')
        if cbow is not None and pca is not None:
            st.caption(f'本詞對 |100D cosine − 2D cosine| = {abs(cbow-pca):.4f}')
        fig = go.Figure(go.Scattergl(x=embedding['coords'][:, 0], y=embedding['coords'][:, 1], mode='markers',
                                    text=embedding['words'], marker={'color': '#8496a5', 'opacity': 0.18, 'size': 4}, name='完整訓練詞表'))
        highlighted = list(dict.fromkeys([w1, w2] + [w for word in [w1, w2] for w, _ in neighbors(word, embedding['vectors'], evocab)[:3]]))
        highlighted = [w for w in highlighted if w in evocab]
        if highlighted:
            points = embedding['coords'][[evocab[w] for w in highlighted]]
            fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], text=highlighted, mode='markers+text', textposition='top center', marker={'size': 10, 'color': '#078b83'}, name='所選詞與鄰居'))
        fig.update_layout(xaxis_title='PC1', yaxis_title='PC2', height=480)
        st.plotly_chart(fig, width='stretch')
        st.caption('所有座標都來自完整訓練詞表的同一次 centered PCA；切換詞對不會重新 fit。可縮放、拖移、指向查看詞。')
    download(ART / 'word_pairs.csv', '下載完整詞對比較 CSV')


def evaluation():
    st.subheader('RQ1–RQ5｜先看問題，再看證據')
    sections = re.split(r'(?m)^## ', report)
    for number in range(1, 6):
        part = next((s for s in sections if s.startswith(f'RQ{number}')), '')
        if part:
            title, _, body = part.partition('\n')
            with st.expander(title, expanded=number == 3):
                st.markdown(body)
    macro = tables['metrics_macro'].rename(columns=METRIC_NAMES)
    st.markdown('**12 組查詢的平均結果**')
    st.dataframe(macro.round(3), hide_index=True, width='stretch')
    chart = macro.melt(id_vars='method', value_vars=['P@10', 'nDCG@10'], var_name='指標', value_name='分數')
    fig = px.bar(chart, x='method', y='分數', color='指標', barmode='group', color_discrete_sequence=['#0a8f85', '#253d60'])
    fig.update_layout(yaxis_range=[0, 1], height=350)
    st.plotly_chart(fig, width='stretch')
    st.caption('名稱含 _dot 的方法是 dot product 對照；其餘為 cosine similarity。SD 描述題目間差異，不是可信區間；各方法不保證每題都贏。')
    q = st.selectbox('查看單題', queries, format_func=lambda q: f"{q['qid']} · {q['query']}", key='eval_query')
    qid = q['qid']
    st.dataframe(tables['metrics'][tables['metrics'].qid == qid].rename(columns=METRIC_NAMES), hide_index=True, width='stretch')
    method = st.selectbox('查看此題前十名的方法', tables['metrics_macro'].method.tolist(), index=4)
    st.dataframe(tables['top10'][(tables['top10'].qid == qid) & (tables['top10'].method == method)], hide_index=True, width='stretch')
    st.write('P@10 的分母固定是 10；Recall@10 的分母是全語料 relevant_count；MRR 使用完整正分排序的第一個命中；nDCG@10 使用 binary relevance 與名次折扣。')
    error = summary['retrieval']['cosine_prior_l2_max_abs_error']
    st.success(f'獨立 raw cosine similarity 與 L2 normalization 後 dot product 的最大誤差：{error:.2e}。數學上相同；浮點同分可能有極小差異。')
    st.info('本次評分規則只檢查關鍵字，容易讓字面相近的方法得高分。1,000 篇又都屬於同一主題，結果不能代表所有文章。這些判定尚未人工逐篇核對。')
    with st.expander('檢查每個 label、規則與原文證據'):
        st.json(q)
        labels = tables['qrels'][tables['qrels'].qid == qid]
        st.dataframe(labels, hide_index=True, width='stretch')
        st.caption('label_source 保留規則或人工覆寫的來源；相關與不相關的全部判定均可下載。')
    st.markdown('**人工檢查後如何覆寫**')
    st.code('qid,pmid,relevant,label_source\nq01,39734391,0,manual_review\n\npython -m experiments --overrides labels.csv', language='text')
    st.caption('上列只是覆寫格式範例，不代表這筆 label 已完成審閱。重新執行後重啟 app，以重新載入 artifacts。')
    for name in ['qrels.csv', 'metrics.csv', 'metrics_macro.csv', 'top10.csv']:
        download(ART / name)
    download(ROOT / 'report.md', '下載完整研究報告')
    with st.expander('學生反思草稿｜請本人審閱'):
        reflection = report.split('## 結論與下一步', 1)[-1].split('## 技術細節', 1)[0]
        st.markdown(reflection)


st.markdown('## GLP-1 Term Weighting Lab')
st.markdown('**Biomedical IR · Homework 3**　｜　TF-IDF 主實驗 × CBOW word embedding 對照')
st.sidebar.markdown('### 研究導覽')
section = st.sidebar.radio('選擇實驗', SECTIONS, label_visibility='collapsed')
st.sidebar.caption('同一份語料 · 同一組規則\n\n先觀察數字，再回到原文確認。')
try:
    artifact_version = tuple((str(p), p.stat().st_mtime_ns) for p in [ART / 'summary.json', ROOT / 'report.md', *sorted(ART.glob('*.csv'))])
    summary, queries, tables, report = load_artifacts(artifact_version)
    required = {'metrics', 'metrics_macro', 'qrels', 'term_statistics', 'tf_demo', 'top10', 'word_pairs'}
    if not required.issubset(tables):
        raise FileNotFoundError('Incomplete CSV artifacts')
    index = load_index()
    embedding = load_embedding()
except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
    st.error('資料或實驗 artifacts 尚未完整準備。請在專案根目錄執行下列指令，再重新啟動 app。')
    st.code('python -m experiments', language='bash')
    st.caption(str(exc))
    st.stop()

{SECTIONS[0]: overview, SECTIONS[1]: search, SECTIONS[2]: microscope,
 SECTIONS[3]: word_similarity, SECTIONS[4]: evaluation}[section]()
