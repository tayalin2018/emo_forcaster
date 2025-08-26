# train_gru.py
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from src.load_data import load_timelines
from src.splits import louo_splits
from src.data.sequence_dataset import SeqDataset, collate_batch
from src.models.gru_forecaster import GRUForecasterTime

# Hyperparams
K = 5
H = 1
BS = 12
EPOCHS = 4          # fewer epochs; encoder is frozen
LR = 2e-4
CLIP = 1.0

def evaluate_fold(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # safe spearman (avoid warnings on tiny folds)
    if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        rho = np.nan
    else:
        rho = spearmanr(y_true, y_pred).statistic
    return mae, rmse, rho

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")

def main():
    os.makedirs("reports", exist_ok=True)
    df = load_timelines("data/raw")
    device = pick_device()
    print(f"Loaded {len(df)} posts / {df.timeline_id.nunique()} timelines. Using device: {device}")

    results, pred_rows = [], []

    for tl, tr_df, te_df in louo_splits(df):
        tr_ds = SeqDataset(tr_df, k=K)
        te_ds = SeqDataset(te_df, k=K)
        if len(te_ds) == 0 or len(tr_ds) == 0:
            continue

        tr_dl = DataLoader(tr_ds, batch_size=BS, shuffle=True,  collate_fn=collate_batch)
        te_dl = DataLoader(te_ds, batch_size=BS, shuffle=False, collate_fn=collate_batch)

        model = GRUForecasterTime(k=K, h=H).to(device)
        # >>> only train unfrozen params (GRU, fuse, head, time2vec)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=LR)
        loss_fn = nn.L1Loss()

        print(f"[{tl}] train {len(tr_ds)} samples, test {len(te_ds)} samples")

        # train
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss, n_items = 0.0, 0
            for texts, times, y, _, _ in tr_dl:
                y = y.to(device)
                pred = model(texts, times).view(-1)
                loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(trainable, CLIP)
                opt.step()
                # stats
                epoch_loss += loss.item() * y.size(0)
                n_items += y.size(0)
            print(f"[{tl}] epoch {epoch+1}/{EPOCHS} - train MAE {epoch_loss / max(1,n_items):.4f}")

        # eval
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for texts, times, y, tids, tidx in te_dl:
                pred = model(texts, times).view(-1).cpu().numpy().tolist()
                y_np = y.cpu().numpy().tolist()
                for p, g, t_id, t_ix in zip(pred, y_np, tids, tidx):
                    pred_rows.append({
                        "timeline_id": t_id,
                        "target_post_idx": int(t_ix),
                        "wb_true": float(g),
                        "wb_pred": float(np.clip(p, 1, 10)),
                        "heldout_user": tl
                    })
                    preds.append(p); gold.append(g)

        preds = np.clip(np.array(preds, dtype=float), 1, 10).tolist()
        mae, rmse, rho = evaluate_fold(gold, preds)
        results.append({"heldout_user": tl, "N_test": len(gold),
                        "gru_mae": mae, "gru_rmse": rmse, "gru_rho": rho})
        print(f"[{tl}] MAE {mae:.3f} | RMSE {rmse:.3f} | rho {rho if rho==rho else float('nan'):.3f}")

    # save reports
    if results:
        pd.DataFrame(results).sort_values("heldout_user").to_csv("reports/gru_louo.csv", index=False)
        pd.DataFrame(pred_rows).to_csv("reports/gru_preds.csv", index=False)
        print("Saved: reports/gru_louo.csv and reports/gru_preds.csv")
    else:
        print("No folds produced results — check K or your data.")

if __name__ == "__main__":
    main()
