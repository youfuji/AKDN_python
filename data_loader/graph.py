from typing import Tuple
import torch

from .interactions import InteractionData
from .kg import KGData

class GraphBuilder:
    """
    InteractionData + KGData を受け取り、
    - LightGCN 用の edge_index, edge_norm
    - AKDN 用の item_kg_neighbors
    をまとめて返すヘルパー。
    KGAT を追加したくなったら、このクラスに KGAT 用の
    隣接行列やノードIDマッピングを追加すればOK。
    """

    def __init__(self, inter_data: InteractionData, kg_data: KGData):
        self.inter_data = inter_data
        self.kg_data = kg_data

        self.num_users = inter_data.num_users
        self.num_items = inter_data.num_items

        # IG: LightGCN 形式
        self.edge_index, self.edge_norm = self._build_interaction_graph()

        # KG: AKDN 形式の item_kg_neighbors をそのまま公開
        self.item_kg_neighbors = kg_data.item_kg_neighbors
        self.entity_kg_neighbors = kg_data.entity_neighbors

    def _build_interaction_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN の 1/sqrt(|N_u||N_i|) 正規化付きで
        edge_index と edge_norm を作る。
        edge_index[0] が user_id, edge_index[1] が item_id。
        （後で KGAT ではここから別の表現を作ればよい）
        """

        user_ids = []
        item_ids = []
        for (u, i) in self.inter_data.interactions:
            user_ids.append(u)
            item_ids.append(i)

        u = torch.tensor(user_ids, dtype=torch.long)
        i = torch.tensor(item_ids, dtype=torch.long)

        edge_index = torch.stack([u, i], dim=0)  # (2, E)

        user_deg = torch.zeros(self.num_users, dtype=torch.float)
        item_deg = torch.zeros(self.num_items, dtype=torch.float)
        for (uu, ii) in self.inter_data.interactions:
            user_deg[uu] += 1.0
            item_deg[ii] += 1.0

        norm = 1.0 / (torch.sqrt(user_deg[u]) * torch.sqrt(item_deg[i]))
        return edge_index, norm
