# AKDN_python
Reproduce the model AKDN from this paper（https://ieeexplore.ieee.org/document/10786928）

# Data
Please download Yelp2018 dataset from:
https://github.com/xiangwang1223/knowledge_graph_attention_network
and place it under `data/yelp2018/`.

```
AKDN_PYTHON/
  data/
    amazon-book/
    last-fm/
    yelp2018/
  data/
    __init__.py
    interactions.py   # train.txt 読み込み (user-item)
    kg.py             # kg_final.txt 読み込み (KG triple)
    graph.py          # IG の edge_index / norm, KG neighbors を構築
  sampling/
    __init__.py
    bpr.py            # BPRSampler (u,i,j) を作る
  models/
    __init__.py
    akdn.py           # AKDN 本体（KGAttention + LightGCN + Gate）
  train_akdn.py       # 学習スクリプト（1エポック学習ループ）
  recommend_akdn.py   # 学習済みモデルで推薦するスクリプト
```