import torch
from torch.utils.data import Dataset

def _make_seq_windows(df_user, k=5):
    """
    For ONE user (sorted by post_idx), create samples:
      texts:  [k] strings (last k posts)
      times:  [k] floats in [0,1] (normalized post index)
      target: float (next post's well-being)
      meta:   timeline_id, target_idx (the index of the next post)
    """
    samples = []
    g = df_user.sort_values("post_idx").reset_index(drop=True)
    T = len(g)
    if T < k + 1:
        return samples
    tl_id = g.loc[0, "timeline_id"]
    max_idx = max(1, T - 1)
    # window end t predicts t+1
    for t in range(k - 1, T - 1):
        texts  = [g.loc[t - (k - 1 - i), "text"] for i in range(k)]
        times  = [g.loc[t - (k - 1 - i), "post_idx"] / max_idx for i in range(k)]
        target = float(g.loc[t + 1, "wb"])
        target_idx = int(g.loc[t + 1, "post_idx"])
        samples.append({
            "texts": texts,
            "times": torch.tensor(times, dtype=torch.float32),
            "y": torch.tensor(target, dtype=torch.float32),
            "timeline_id": tl_id,
            "target_idx": target_idx
        })
    return samples

class SeqDataset(Dataset):
    def __init__(self, df, k=5):
        self.k = k
        self.samples = []
        for _, g in df.groupby("timeline_id"):
            self.samples.extend(_make_seq_windows(g, k=k))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_batch(batch):
    texts = [b["texts"] for b in batch]                         # list of [k] strings
    times = torch.stack([b["times"] for b in batch], dim=0)     # [B, k]
    y     = torch.stack([b["y"]     for b in batch], dim=0)     # [B]
    tids  = [b["timeline_id"] for b in batch]                   # list[str]
    tidx  = [b["target_idx"]  for b in batch]                   # list[int]
    return texts, times, y, tids, tidx
