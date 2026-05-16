import torch


class DistMult(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int):
        super().__init__()
        self.ent = torch.nn.Embedding(num_entities, emb_dim)
        self.rel = torch.nn.Embedding(num_relations, emb_dim)
        torch.nn.init.xavier_uniform_(self.ent.weight)
        torch.nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h, r, t):
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(dim=-1)
