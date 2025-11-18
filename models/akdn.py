import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class KGAttentionLayer(nn.Module):
    """
    AKDN の KG attention (式(1),(2)) に対応する簡略版レイヤー。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W_k = nn.Linear(2 * dim, dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, item_emb, entity_emb, relation_emb, item_kg_neighbors):
        """
        item_emb: (num_items, d)
        entity_emb: (num_entities, d)
        relation_emb: (num_relations, d)
        item_kg_neighbors: Dict[item_id, List[(r, v)]]
        """
        num_items, d = item_emb.size()
        device = item_emb.device
        out = torch.zeros_like(item_emb)

        for i in range(num_items):
            neighbors = item_kg_neighbors.get(i, [])
            if not neighbors:
                out[i] = item_emb[i]
                continue

            rel_ids = torch.tensor([r for (r, v) in neighbors],
                                   dtype=torch.long, device=device)
            ent_ids = torch.tensor([v for (r, v) in neighbors],
                                   dtype=torch.long, device=device)

            r_vec = relation_emb[rel_ids]    # (n, d)
            v_vec = entity_emb[ent_ids]      # (n, d)
            i_vec = item_emb[i].expand_as(v_vec)

            hv = v_vec * i_vec               # e_v ⊙ e_i
            concat = torch.cat([hv, hv], dim=-1)  # 2d 次元（簡易）

            att_raw = r_vec * self.W_k(concat)    # (n, d)
            att_raw = self.leaky_relu(att_raw.sum(dim=-1))  # (n,)
            alpha = F.softmax(att_raw, dim=0)              # (n,)

            out[i] = torch.sum(alpha.unsqueeze(-1) * v_vec, dim=0)

        return out


class LightGCNLayer(nn.Module):
    """
    Interaction Graph 上での LightGCN 更新。
    edge_index[0]: user_ids, edge_index[1]: item_ids
    """

    def __init__(self, num_users, num_items, edge_index, edge_norm):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.edge_index = edge_index
        self.edge_norm = edge_norm

    def forward(self, user_emb, item_emb):
        u, i = self.edge_index
        norm = self.edge_norm.unsqueeze(-1)

        msg_to_items = norm * user_emb[u]         # (E, d)
        agg_items = scatter_add(msg_to_items, i,
                                dim=0, dim_size=self.num_items)

        msg_to_users = norm * item_emb[i]
        agg_users = scatter_add(msg_to_users, u,
                                dim=0, dim_size=self.num_users)

        return agg_users, agg_items


class AKDNLayer(nn.Module):
    """
    1 層ぶんの AKDN:
    - KGAttentionLayer で KG item 表現
    - LightGCNLayer で CF user/item 表現
    - Gate で item を融合
    """

    def __init__(self, dim, num_users, num_items, edge_index, edge_norm):
        super().__init__()
        self.kg_layer = KGAttentionLayer(dim)
        self.lgcn_layer = LightGCNLayer(num_users, num_items,
                                        edge_index, edge_norm)
        self.W_a = nn.Linear(dim, dim)
        self.W_b = nn.Linear(dim, dim)

    def forward(self, user_emb, item_emb,
                entity_emb, relation_emb, item_kg_neighbors):
        kg_item_emb = self.kg_layer(item_emb, entity_emb,
                                    relation_emb, item_kg_neighbors)
        cf_user_emb, cf_item_emb = self.lgcn_layer(user_emb, item_emb)

        gate = torch.sigmoid(self.W_a(kg_item_emb) + self.W_b(cf_item_emb))
        fused_item_emb = gate * kg_item_emb + (1.0 - gate) * cf_item_emb

        new_user_emb = cf_user_emb
        new_item_emb = fused_item_emb
        return new_user_emb, new_item_emb, entity_emb


class AKDN(nn.Module):
    """
    AKDN 全体モデル。
    - propagate(): e_u^*, tilde e_i^* を返す
    - forward(u, pos_i, neg_i): BPR loss を返す
    """

    def __init__(self,
                 num_users, num_items,
                 num_entities, num_relations,
                 edge_index, edge_norm,
                 item_kg_neighbors,
                 dim=64, num_layers=2, reg=1e-4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.reg = reg
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        self.layers = nn.ModuleList([
            AKDNLayer(dim, num_users, num_items,
                      edge_index, edge_norm)
            for _ in range(num_layers)
        ])

        self.item_kg_neighbors = item_kg_neighbors

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.entity_emb.weight, std=0.01)
        nn.init.normal_(self.relation_emb.weight, std=0.01)

    def propagate(self):
        all_user_embs = [self.user_emb.weight]
        all_cf_item_embs = [self.item_emb.weight]

        u = self.user_emb.weight
        i = self.item_emb.weight
        v = self.entity_emb.weight
        r = self.relation_emb.weight

        for layer in self.layers:
            u, i, v = layer(u, i, v, r, self.item_kg_neighbors)
            all_user_embs.append(u)
            all_cf_item_embs.append(i)

        final_user_emb = torch.stack(all_user_embs, dim=0).sum(dim=0)
        final_item_emb = torch.stack(all_cf_item_embs, dim=0).sum(dim=0)
        return final_user_emb, final_item_emb

    def forward(self, user, pos_item, neg_item):
        final_user_emb, final_item_emb = self.propagate()

        u_e = final_user_emb[user]
        pos_e = final_item_emb[pos_item]
        neg_e = final_item_emb[neg_item]

        pos_scores = (u_e * pos_e).sum(dim=-1)
        neg_scores = (u_e * neg_e).sum(dim=-1)

        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        l2 = (u_e.norm(2).pow(2) +
              pos_e.norm(2).pow(2) +
              neg_e.norm(2).pow(2)) / user.shape[0]
        return bpr_loss + self.reg * l2

    def get_user_item_scores(self):
        final_user_emb, final_item_emb = self.propagate()
        return final_user_emb @ final_item_emb.t()  # (U, I)
