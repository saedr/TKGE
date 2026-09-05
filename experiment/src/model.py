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

    def score_tail_batch(self, h, r, candidates):
        query = self.ent(h) * self.rel(r)
        return query @ self.ent(candidates).transpose(0, 1)


class TransE(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int, p: int = 2):
        super().__init__()
        self.ent = torch.nn.Embedding(num_entities, emb_dim)
        self.rel = torch.nn.Embedding(num_relations, emb_dim)
        self.p = p
        torch.nn.init.xavier_uniform_(self.ent.weight)
        torch.nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h, r, t):
        return -(self.ent(h) + self.rel(r) - self.ent(t)).norm(p=self.p, dim=-1)

    def score_tail_batch(self, h, r, candidates):
        query = self.ent(h) + self.rel(r)
        return -torch.cdist(query, self.ent(candidates), p=self.p)


class ComplEx(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int):
        super().__init__()
        self.ent_re = torch.nn.Embedding(num_entities, emb_dim)
        self.ent_im = torch.nn.Embedding(num_entities, emb_dim)
        self.rel_re = torch.nn.Embedding(num_relations, emb_dim)
        self.rel_im = torch.nn.Embedding(num_relations, emb_dim)
        torch.nn.init.xavier_uniform_(self.ent_re.weight)
        torch.nn.init.xavier_uniform_(self.ent_im.weight)
        torch.nn.init.xavier_uniform_(self.rel_re.weight)
        torch.nn.init.xavier_uniform_(self.rel_im.weight)

    def score(self, h, r, t):
        h_re = self.ent_re(h)
        h_im = self.ent_im(h)
        r_re = self.rel_re(r)
        r_im = self.rel_im(r)
        t_re = self.ent_re(t)
        t_im = self.ent_im(t)
        return (
            h_re * r_re * t_re
            + h_im * r_re * t_im
            + h_re * r_im * t_im
            - h_im * r_im * t_re
        ).sum(dim=-1)

    def score_tail_batch(self, h, r, candidates):
        h_re = self.ent_re(h)
        h_im = self.ent_im(h)
        r_re = self.rel_re(r)
        r_im = self.rel_im(r)
        coeff_re = h_re * r_re - h_im * r_im
        coeff_im = h_im * r_re + h_re * r_im
        return (
            coeff_re @ self.ent_re(candidates).transpose(0, 1)
            + coeff_im @ self.ent_im(candidates).transpose(0, 1)
        )
