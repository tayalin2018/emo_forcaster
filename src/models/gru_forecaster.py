# src/models/gru_forecaster.py
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class Time2Vec(nn.Module):
    def __init__(self, d_time=9):  # 1 linear + 8 sinusoidal
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(d_time - 1))
        self.b  = nn.Parameter(torch.zeros(d_time - 1))

    def forward(self, t):          # t: [B, K] in [0,1]
        t = t.unsqueeze(-1)        # [B,K,1]
        v1 = self.w0 * t           # [B,K,1]
        v2 = torch.sin(t * self.w + self.b)  # [B,K,d_time-1] (broadcast)
        return torch.cat([v1, v2], dim=-1)   # [B,K,d_time]

class GRUForecasterTime(nn.Module):
    def __init__(self, enc_id="sentence-transformers/all-MiniLM-L6-v2",
                 k=5, h=1, d_time=9, dropout=0.1):
        super().__init__()
        self.k, self.h = k, h
        self.tok = AutoTokenizer.from_pretrained(enc_id)
        self.enc = AutoModel.from_pretrained(enc_id)
        # >>> Freeze the transformer encoder <<<
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

        d_txt = self.enc.config.hidden_size
        self.t2v = Time2Vec(d_time=d_time)
        self.fuse = nn.Linear(d_txt + d_time, d_txt)
        self.gru  = nn.GRU(d_txt, d_txt, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_txt, h)

    def _encode_step(self, texts):
        dev = next(self.parameters()).device
        toks = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(dev)
        # >>> No gradients for the frozen encoder <<<
        with torch.no_grad():
            out = self.enc(**toks).last_hidden_state[:, 0]   # [CLS]-like pooling
        return out  # [B, d_txt]

    def forward(self, batch_ctx_texts, batch_ctx_times):
        """
        batch_ctx_texts: list of [k] strings per item (len B)
        batch_ctx_times: tensor [B, k] floats in [0,1]
        """
        B, K = len(batch_ctx_texts), len(batch_ctx_texts[0])
        assert K == self.k, "k mismatch"

        # encode text for each position in the window
        embs = []
        for i in range(K):
            step_texts = [ctx[i] for ctx in batch_ctx_texts]    # list len B
            embs.append(self._encode_step(step_texts))          # [B,d]
        Xtxt = torch.stack(embs, dim=1)                         # [B,K,d]

        # time features
        Xt   = self.t2v(batch_ctx_times.to(Xtxt.device))        # [B,K,d_time]

        # fuse text + time
        X = torch.cat([Xtxt, Xt], dim=-1)                       # [B,K,d+d_time]
        X = self.fuse(X)                                        # [B,K,d]
        X = self.drop(X)

        _, h = self.gru(X)                                      # h: [1,B,d]
        h = self.drop(h[-1])                                    # [B,d]
        y = self.head(h)                                        # [B,h]
        return y.squeeze(-1)
