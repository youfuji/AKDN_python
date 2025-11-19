import torch
from typing import List

from data_loader.interactions import InteractionData
from data_loader.kg import KGData
from data_loader.graph import GraphBuilder
from models.akdn import AKDN

def recommend_for_user(model: AKDN,
                       inter_data: InteractionData,
                       user_id: int,
                       topk: int = 10,
                       device: str = "cuda") -> List[int]:
    model.eval()
    with torch.no_grad():
        model.to(device)
        final_user_emb, final_item_emb = model.propagate()
        u_vec = final_user_emb[user_id]                # (d,)
        scores = final_item_emb @ u_vec                # (I,)

        for i in inter_data.user_pos_items[user_id]:
            scores[i] = -1e9

        _, top_items = torch.topk(scores, k=topk)
        return top_items.cpu().tolist()

# 使い方イメージ：
# 1. train_akdn.py で学習＆保存
# 2. ここで load_state_dict して上の関数を呼ぶ
