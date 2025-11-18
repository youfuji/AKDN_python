from collections import defaultdict
from typing import List, Tuple, Dict

class InteractionData:
    """
    train.txt (user item) を読み込んで、
    - interactions: List[(u, i)]
    - user_pos_items: Dict[u, set(i)]
    - user_neighbors_items: Dict[u, List[i]]
    - item_neighbors_users: Dict[i, List[u]]
    を保持する小さなクラス
    """

    def __init__(self, path: str):
        self.interactions: List[Tuple[int, int]] = []
        self.user_pos_items: Dict[int, set] = defaultdict(set)
        self.user_neighbors_items: Dict[int, List[int]] = defaultdict(list)
        self.item_neighbors_users: Dict[int, List[int]] = defaultdict(list)

        self._load(path)

    def _load(self, path: str):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                u_str, i_str = line.split()
                u, i = int(u_str), int(i_str)

                self.interactions.append((u, i))
                self.user_pos_items[u].add(i)
                self.user_neighbors_items[u].append(i)
                self.item_neighbors_users[i].append(u)

    @property
    def num_users(self) -> int:
        return max(u for (u, _) in self.interactions) + 1

    @property
    def num_items(self) -> int:
        return max(i for (_, i) in self.interactions) + 1
