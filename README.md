# GLP-1 詞項權重研究室

**Biomedical IR Homework 3 · Biomedical Information Retrieval Homework 3**

以提供的 1,000 篇 PubMed GLP-1 文獻，回答 RQ1～RQ5。主實驗研究 TF、Log TF、IDF、TF-IDF、L2 與搜尋評估；另以同份語料訓練 CBOW，對照字詞分布相似度與 PCA 二維投影。

## 本機啟動

Python 3.13，首次安裝：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m experiments
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m streamlit run app.py
```

已包含 `artifacts/` 的實驗輸出，展示時不必重新訓練模型，也不需 API 金鑰。瀏覽器開啟 http://localhost:8501。

## 繳交內容

- `app.py`：可操作的 Streamlit 展示。
- `ir_core.py`：前處理、權重、搜尋、相似度與評估指標。
- `experiments.py`：重現實驗與 CBOW 訓練。
- `report.md`：完整 RQ 回答、實際數據、原理解說與反思草稿。
- `artifacts/`：可追溯的查詢、相關性判斷、排名、指標與模型向量。

## 解讀原則

TF-IDF 文件搜尋與「字詞欄向量的 cosine」是不同實驗；後者的 IDF 倍數會抵消。CBOW 相似度代表上下文關聯，不能直接證明兩字同義。PCA 把 100 維投影到 2 維，會丟掉資訊，因此原始 100 維分數是主要比較依據。

評估資料若標示為 silver，表示由事先固定的主題規則產生，具有詞面偏差，並非人工黃金答案。報告中的第一人稱反思是可修改的草稿，請確認符合自己理解後再繳交。

## 方法參考

- [scikit-learn TF-IDF 公式與normalization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
- [Gensim CBOW／Word2Vec 參數與重現性](https://radimrehurek.com/gensim/models/word2vec.html)
- [scikit-learn PCA 與解釋變異比例](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Streamlit Community Cloud 官方部署步驟](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)



