import numpy as np
import torch


def train_distmult(model, triples, num_entities, cfg, device):
    model.to(device)
    epochs = cfg["training"]["epochs"]
    bs = cfg["training"]["batch_size"]
    neg_k = cfg["training"]["negatives_per_positive"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    arr = np.array(triples, dtype=np.int64)
    losses = []
    for _ in range(epochs):
        np.random.shuffle(arr)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(arr), bs):
            batch = arr[i : i + bs]
            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)
            pos = model.score(h, r, t)

            nh = h.repeat_interleave(neg_k)
            nr = r.repeat_interleave(neg_k)
            nt = torch.randint(0, num_entities, (len(h) * neg_k,), device=device)
            neg = model.score(nh, nr, nt)

            loss = torch.nn.functional.softplus(-pos).mean() + torch.nn.functional.softplus(neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        losses.append(ep_loss / max(1, n_batches))
    return losses
