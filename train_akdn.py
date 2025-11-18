import torch
from torch.optim import Adam

from akdn_project.data.interactions import InteractionData
from akdn_project.data.kg import KGData
from akdn_project.data.graph import GraphBuilder
from akdn_project.sampling.bpr import BPRSampler
from akdn_project.models.akdn import AKDN

def main():
    train_path = "data/yelp2018/train.txt"
    kg_path = "data/yelp2018/kg_final.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inter_data = InteractionData(train_path)
    kg_data = KGData(kg_path, num_items=inter_data.num_items)
    graph = GraphBuilder(inter_data, kg_data)

    model = AKDN(
        num_users=inter_data.num_users,
        num_items=inter_data.num_items,
        num_entities=kg_data.num_entities,
        num_relations=kg_data.num_relations,
        edge_index=graph.edge_index.to(device),
        edge_norm=graph.edge_norm.to(device),
        item_kg_neighbors=graph.item_kg_neighbors,
        dim=64,
        num_layers=2,
        reg=1e-4,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    sampler = BPRSampler(
        interactions=inter_data,
        batch_size=1024,
        num_items=inter_data.num_items,
        num_negative=1,
    )

    for epoch in range(50):
        model.train()
        total_loss, num_batches = 0.0, 0
        for u, i, j in sampler:
            u, i, j = u.to(device), i.to(device), j.to(device)
            optimizer.zero_grad()
            loss = model(u, i, j)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"[Epoch {epoch}] loss = {total_loss / max(1, num_batches):.4f}")

if __name__ == "__main__":
    main()
