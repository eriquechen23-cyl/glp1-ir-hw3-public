"""Generate a direct, student-readable RQ report from measured artifacts."""
from pathlib import Path

def build_report(index, macro, metrics, pairs, summary):
    m={r['method']:r for r in macro}
    t={r['term']:r for r in summary['term_statistics']}
    by_query={(r['qid'],r['method']):r for r in metrics}
    first_hits=sum(r['method']=='log_tfidf' and r['mrr']==1 for r in metrics)
    query_count=sum(r['method']=='log_tfidf' for r in metrics)
    def compare(a,b):
        return '較高' if a>b+1e-12 else ('較低' if a<b-1e-12 else '相同')
    top_p=max(('tf','log_tf','idf','tfidf','log_tfidf'),key=lambda k:m[k]['precision_at_10'])
    top_n=max(('tf','log_tf','idf','tfidf','log_tfidf'),key=lambda k:m[k]['ndcg_at_10'])
    names={'tf':'TF','log_tf':'Log TF','idf':'IDF-only','tfidf':'TF-IDF','log_tfidf':'Log TF-IDF'}
    override_count=summary['queries'].get('override_count',0)

    lines=[
    '# TF-IDF 能不能改善 GLP-1 文獻搜尋？',
    '',
    '**Biomedical IR｜Homework 3｜Biomedical Information Retrieval Homework 3**',
    '',
    '這份報告比較不同的文字計分方式，觀察哪些方式比較容易把相關文章排在前面。每一題依序說明：做了什麼、得到什麼結果、我的看法。所有數字都來自同一份資料與同一組查詢。',
    '',
    '## 這次實驗怎麼做',
    '',
    '1. 使用提供的 1,000 篇 GLP-1 文獻，讀取標題和摘要。',
    '2. 固定 12 組英文查詢，例如 semaglutide weight loss、kidney disease renal outcomes。',
    '3. 比較 TF、Log TF、IDF-only、TF-IDF、Log TF-IDF。除了指定的對照實驗，其餘都用 cosine similarity 排名。',
    '4. 每組查詢取前 10 篇，檢查有多少篇符合事先固定的關鍵字規則，並記錄名次。',
    '',
    '**評分依據：** 這份資料沒有現成的人工正確答案。因此先以固定的關鍵字規則列出每題的參考文章，稱為 silver labels。這些規則沒有依搜尋結果調整，但尚未由人工逐篇確認；後面的分數只表示符合這套規則的程度。',
    '',
    '## RQ1：TF 和 Log TF，會怎麼影響結果？',
    '',
    '**做法：** TF 直接計算一個字在文章出現幾次。Log TF 也看出現次數，但會減少一直重複用字所增加的分數。兩者使用相同的資料、查詢和 cosine similarity。',
    '',
    f"**結果：** 在 12 題中，TF 的前 10 篇平均有 {m['tf']['precision_at_10']*10:.1f} 篇符合評分規則；改成 Log TF 後，為 {m['log_tf']['precision_at_10']*10:.1f} 篇。實際例子是 PMID 32390511：glp1 出現 28 次，TF=28，但 Log TF=4.332。",
    '',
    '**我的看法：** 同一個字寫很多次，不代表文章一定更符合搜尋問題。上面的平均篇數可以讓我們檢查降低重複用字影響後，結果是否改善，而不是只靠直覺決定。',
    '',
    '## RQ2：為什麼需要 IDF？',
    '',
    '**做法：** IDF 根據「有多少篇文章出現這個字」給分。很多文章都有的字，給的分數較低；只有少數文章有的字，分數較高。',
    '',
    f"**結果：** glp1 出現在 {t['glp1']['df']} 篇文章，IDF={t['glp1']['idf']:.3f}；pancreatitis 只出現在 {t['pancreatitis']['df']} 篇，IDF={t['pancreatitis']['idf']:.3f}。搜尋 pancreatitis 時，這個字比幾乎每篇都有的 glp1 更能幫忙縮小範圍。",
    '',
    '**我的看法：** 我們要找的是能區分文章的字，而不是每篇都有的字。不過少見的字也可能是拼字錯誤，所以 IDF 高不代表文章內容一定更重要。',
    '',
    '## RQ3：TF-IDF 比只用 TF 或 IDF 更好嗎？',
    '',
    '**做法：** 固定 12 題查詢與評分規則，只改變文字的計分方式。下表先用「前 10 篇平均符合幾篇」來比較。',
    '',
    '| 方法 | 前 10 篇平均符合幾篇 | P@10 | nDCG@10 |',
    '|---|---:|---:|---:|']
    for key,label in [('tf','TF'),('log_tf','Log TF'),('idf','IDF-only'),('tfidf','TF-IDF'),('log_tfidf','Log TF-IDF')]:
        r=m[key];lines.append(f"| {label} | {r['precision_at_10']*10:.2f} | {r['precision_at_10']:.3f} | {r['ndcg_at_10']:.3f} |")
    lines += [
    '',
    f"**結果：** 以本次評分依據計算，TF-IDF 的 P@10 比 TF {compare(m['tfidf']['precision_at_10'],m['tf']['precision_at_10'])}。五種 cosine 方法中，P@10 最高是 {names[top_p]}，nDCG@10 最高是 {names[top_n]}。兩個指標分別看命中數和前後排名。",
    '',
    f"**我的看法：** TF-IDF 同時考慮這篇提到幾次，以及這個字有多普遍，但不能預設它一定最好。例如 liraglutide obesity 這題，Log TF 的 P@10={by_query[('q02','log_tf')]['precision_at_10']:.3f}，Log TF-IDF={by_query[('q02','log_tfidf')]['precision_at_10']:.3f}。應該同時看整體表現和個別查詢。",
    '',
    '**評分規則也有影響：** 查詢 semaglutide weight loss 時，第 1 名是 PMID 39734391，標題為 Semaglutide-Associated Gastric Pneumatosis。它提到用藥減重，依原始自動規則會被列入參考文章（人工覆寫後以修訂名單為準）；但主要內容是在談不良反應，不一定是使用者想看的減重效果研究。這說明分數高，不一定等於實際搜尋需求已滿足。',
    '',
    '## RQ4：需要做 L2 normalization 嗎？',
    '',
    '**做法：** 如果直接把字詞分數相乘再相加（dot product），文章可能因為較長、關鍵字重複較多而得高分。L2 normalization 是先調整整份分數的大小，讓比較比較不受這種影響。這裡比較未調整的 dot product 和調整後的 cosine similarity。',
    '',
    '| 權重 | 未做 L2 normalization：P@10 | cosine similarity：P@10 |',
    '|---|---:|---:|',
    f"| TF-IDF | {m['tfidf_dot']['precision_at_10']:.3f} | {m['tfidf']['precision_at_10']:.3f} |",
    f"| Log TF-IDF | {m['log_tfidf_dot']['precision_at_10']:.3f} | {m['log_tfidf']['precision_at_10']:.3f} |",
    '',
    f"**結果：** TF-IDF 的 cosine P@10 比原始 dot product {compare(m['tfidf']['precision_at_10'],m['tfidf_dot']['precision_at_10'])}；Log TF-IDF 的 cosine P@10 則{compare(m['log_tfidf']['precision_at_10'],m['log_tfidf_dot']['precision_at_10'])}。另有一個必須分清楚的地方：直接計算 cosine similarity，和先做 L2 normalization 再算 dot product，是同一件事。本次排名相同，分數誤差只有 {summary['retrieval']['cosine_prior_l2_max_abs_error']:.2e}。",
    '',
    '**我的看法：** L2 normalization 可以減少文章長度的影響，但不保證評分一定提高。我會先用 cosine similarity 作為清楚、穩定的比較基準，再用實驗判斷是否需要其他方式。不能把已經使用 cosine 的結果，再說成因為額外做了 L2 才改善。',
    '',
    '## RQ5：怎麼判斷搜尋結果好不好？',
    '',
    '**做法：** 同時看幾個不同問題，不只用一個分數。',
    '',
    '- **P@10：** 前 10 篇中，有幾篇符合評分規則？例如 8 篇符合就是 0.8。',
    '- **MRR：** 第一篇符合規則的文章排第幾？第 1 名就是 1，第 2 名就是 0.5。',
    '- **nDCG@10：** 符合規則的文章，是否被排在較前面？越接近 1 越好。',
    '- **Recall@10：** 所有參考文章中，前 10 名找回了多少比例？',
    '',
    f"**結果：** Log TF-IDF 的 P@10={m['log_tfidf']['precision_at_10']:.3f}、MRR={m['log_tfidf']['mrr']:.3f}、nDCG@10={m['log_tfidf']['ndcg_at_10']:.3f}。也就是前 10 篇平均約 {m['log_tfidf']['precision_at_10']*10:.1f} 篇符合規則，{query_count} 題中有 {first_hits} 題的第 1 名符合規則。",
    '',
    f"**我的看法：** 第一名找對，不代表整頁都找對。例如 liver steatosis glp1 使用 TF 時，MRR={by_query[('q11','tf')]['mrr']:.3f}，P@10={by_query[('q11','tf')]['precision_at_10']:.3f}。這兩個指標回答不同問題，應同時檢查。",
    '',
    '## WORD1／WORD2：TF-IDF 和 CBOW 看的是同一種相似嗎？',
    '',
    '**TF-IDF 的做法：** 看兩個字是否常出現在相同的文章。每個字都有 1,000 個分數，分別對應 1,000 篇文章，再比較這兩組分數。',
    '',
    '**CBOW 的做法：** 看每個字前後常接哪些字。訓練時，用附近的字猜中間的字；反覆練習後，每個字得到一組 100 個數字，稱為 word embedding。這次使用同一批 1,000 篇文章訓練，再比較兩個字的 100D cosine similarity。',
    '',
    '**PCA 2D 的做法：** 把每個字的 100 個數字縮成 2 個數字，方便畫在平面圖上。但縮減後會失去資訊，因此還要對照原本 100D 的分數。',
    '',
    '| WORD1 | WORD2 | TF-IDF | CBOW 100D | PCA 2D |',
    '|---|---|---:|---:|---:|']
    for r in pairs:
        fmt=lambda v:'未定義' if v is None else f'{v:.3f}'
        lines.append(f"| {r['word1']} | {r['word2']} | {fmt(r['tfidf_column_cosine'])} | {fmt(r['cbow_100d_seed42_cosine'])} | {fmt(r['cbow_pca2d_seed42_cosine'])} |")
    lines += [
    '',
    '**我注意到的差異：**',
    '',
    '1. kidney／renal：TF-IDF=0.267，CBOW100D=0.522。兩字在文章中的出現分布不完全相同，但它們附近使用的字比較接近。',
    '2. weight／loss：TF-IDF=0.777，CBOW100D=0.129。它們常一起出現在文章中，但不表示在句子中能互相替換。',
    '3. agonist／receptor：CBOW100D=-0.068，PCA2D卻變成0.890。本次 PCA 兩軸只保留12.5%的變動資訊，因此不能只因為二維圖上靠近，就判定兩字相似。',
    '',
    '**我的看法：** TF-IDF 和 CBOW 提供不同線索。前者看文章中的分布，後者看前後用字。本作業可以比較它們是否給出一致結果，但這些分數不能直接證明兩個字同義。Streamlit 的相似門檻只是方便觀察，不是經人工驗證的標準。',
    '',
    '**TF-IDF 詞對要注意：** 本實驗比較的是未先做文件 L2 normalization 的字詞分數。固定一個字的 IDF 只是所有分數共同乘上的倍數，計算 cosine similarity 時會抵消，所以這裡的 TF-IDF 詞對分數等於 TF 詞對分數。這和查詢對文章的 TF-IDF 搜尋，是不同的比較。',
    '',
    '## 結論與下一步',
    '',
    f"這份資料與本次評分依據下，Log TF 的 P@10 比 TF {compare(m['log_tf']['precision_at_10'],m['tf']['precision_at_10'])}；Log TF-IDF 比原始 TF-IDF {compare(m['log_tfidf']['precision_at_10'],m['tfidf']['precision_at_10'])}。這是目前題組的觀察，不能推論某個方法在所有資料或查詢都最好。",
    '',
    '如果要繼續改善，我會先人工檢查每題前 10 篇，確認「符合關鍵字規則」是否真的等於「符合搜尋意圖」，再增加更多題目。CBOW 與 PCA 則保留作為詞對分析工具，避免只用一張二維圖下結論。',
    '',
    '以上「我的看法」是供本人審閱、修改的討論草稿；本次沒有宣稱做過逐篇人工標註。',
    '',
    '## 技術細節：供重現與教授核對',
    '',
    '- TF=tf；Log TF=1+ln(tf)，tf=0 時權重為0。',
    '- IDF=ln((N+1)/(DF+1))+1；IDF-only 僅在字詞出現時給 IDF 分數。',
    '- TF-IDF=TF×IDF；Log TF-IDF=Log TF×IDF。',
    '- cosine similarity=(a·b)/(||a||₂ ||b||₂)；零向量或 OOV 不視為已驗證不相似。',
    '- P@10 的分母固定為10；MRR使用完整正分排序；nDCG採binary relevance、log2折扣，與理想前10名比較。',
    '- 英文轉小寫，GLP-1不同寫法合為glp1；不做stemming、不移除stopwords，單獨數字排除；查詢與文件前處理一致。',
    '- CBOW：100D、window=5、min_count=3、epochs=50、negative=10、workers=1；seeds=17/42/73。PCA在完整CBOW詞表上置中，取前兩個方向；不會依使用者輸入的兩個字重做PCA。',
    '- 資料為單一主題的1,000篇英文文獻，不能推論所有醫學資料，也不以文字分數判定療效或安全性。',
    '',
    '重現：`python -m experiments`。人工修訂名單可用 `python -m experiments --overrides labels.csv`，欄位為 `qid,pmid,relevant,label_source`。所有原始查詢、評分依據、排名、分數、詞對來源句子及三個seed的結果，都在 `artifacts/`。',
    '',
    f"資料 SHA-256：`{summary['corpus']['sha256']}`。",
    '',
    '參考：[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)、[CBOW／Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)、[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)。',
    '']
    if override_count:
        lines.insert(4, f'本報告已套用 {override_count} 筆人工覆寫；排名指標及文字數值依修訂後的參考名單計算。')
    return '\n'.join(lines)
