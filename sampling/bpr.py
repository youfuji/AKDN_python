import random
from typing import List, Tuple
import torch

from data_loader.interactions import InteractionData

class BPRSampler:
    """
    BPRSampler = 「正例 (u, i) をベースに、負例 j をランダムサンプルして
    (u, i, j) のバッチを吐き出すイテレータ」

    AKDN でも KGAT でも、BPR 学習ならそのまま共通で使える設計にしてある。
    """

    def __init__(self,
                 interactions: InteractionData,
                 batch_size: int,
                 num_items: int,
                 num_negative: int = 1):
        self.interactions = interactions
        self.batch_size = batch_size
        self.num_items = num_items
        self.num_negative = num_negative

        self.all_pairs: List[Tuple[int, int]] = interactions.interactions

    def __iter__(self):
        random.shuffle(self.all_pairs)

        batch_u, batch_i, batch_j = [], [], []

        for (u, i) in self.all_pairs:
            for _ in range(self.num_negative):
                while True:
                    j = random.randint(0, self.num_items - 1)
                    if j not in self.interactions.user_pos_items[u]:
                        break

                batch_u.append(u)
                batch_i.append(i)
                batch_j.append(j)

                if len(batch_u) == self.batch_size:
                    yield (
                        torch.tensor(batch_u, dtype=torch.long),
                        torch.tensor(batch_i, dtype=torch.long),
                        torch.tensor(batch_j, dtype=torch.long),
                    )
                    batch_u, batch_i, batch_j = [], [], []

        if batch_u:
            yield (
                torch.tensor(batch_u, dtype=torch.long),
                torch.tensor(batch_i, dtype=torch.long),
                torch.tensor(batch_j, dtype=torch.long),
            )
