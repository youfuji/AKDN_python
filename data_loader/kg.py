from collections import defaultdict
from typing import List, Tuple, Dict

class KGData:
    """
    kg_final.txt (h r t) を読み込んで、
    - triples: List[(h, r, t)]
    - item_kg_neighbors: Dict[item_id, List[(r, v)]]
      （AKDNでは「item を head に持つ近傍」だけ使う設計）
    を保持するクラス
    """

    def __init__(self, path: str, num_items: int):
        self.triples: List[Tuple[int, int, int]] = []
        self.item_kg_neighbors: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.entity_neighbors: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        self._load(path, num_items)

    def _load(self, path: str, num_items: int):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                h_str, r_str, t_str = line.split()
                h, r, t = int(h_str), int(r_str), int(t_str)

                self.triples.append((h, r, t))
                self.entity_neighbors[h].append((r, t))

                # AKDN: item を head に持つときだけ利用
                if 0 <= h < num_items:
                    self.item_kg_neighbors[h].append((r, t))

    @property
    def num_entities(self) -> int:
        # h, t の max から entity 数を決める簡易版
        max_ent = -1
        for h, _, t in self.triples:
            max_ent = max(max_ent, h, t)
        return max_ent + 1

    @property
    def num_relations(self) -> int:
        max_rel = -1
        for _, r, _ in self.triples:
            max_rel = max(max_rel, r)
        return max_rel + 1
