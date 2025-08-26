import os, math, numpy as np, pandas as pd

IN_CSV = "reports/gru_preds.csv"
OUT_CSV = "reports/gru_shift_scores.csv"

TAU = 1.0   # change threshold in well-being points
TOL = 1     # allowable index offset when matching events (±1 post)

def change_events(values, tau=1.0, direction="any"):
    """
    Args:
      values: list[float] ordered by target_post_idx
      tau: magnitude threshold for calling a change
      direction: "any" | "down" | "up"
    Returns:
      idxs: list[int] event index positions (relative to values)
      signs: list[int] +1 (up), -1 (down)
    Note:
      Event index is the position of the point *after* the jump.
    """
    v = np.asarray(values, dtype=float)
    d = np.diff(v)
    idxs, signs = [], []
    for i, delta in enumerate(d, start=1):
        sgn = 1 if delta >= 0 else -1
        if abs(delta) >= tau:
            if direction == "down" and sgn != -1: continue
            if direction == "up"   and sgn !=  1: continue
            idxs.append(i)
            signs.append(sgn)
    return idxs, signs

def match_events(pred_idxs, pred_signs, true_idxs, true_signs, tol=1, require_dir=False):
    """
    Greedy one-to-one matching within ±tol. Optionally require direction match.
    Returns: TP, FP, FN, timing_errors(list[int])
    """
    used_true = set()
    tp = 0
    timing = []
    for pi, ps in zip(pred_idxs, pred_signs):
        # find nearest true event not yet used
        best = None
        best_dist = None
        for j, (ti, ts) in enumerate(zip(true_idxs, true_signs)):
            if j in used_true: 
                continue
            if require_dir and (ps != ts):
                continue
            dist = abs(pi - ti)
            if dist <= tol and (best is None or dist < best_dist):
                best, best_dist = j, dist
        if best is not None:
            used_true.add(best)
            tp += 1
            timing.append(best_dist)
    fp = len(pred_idxs) - tp
    fn = len(true_idxs) - tp
    return tp, fp, fn, timing

def prf1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if (tp==0 and fp==0 and fn==0) else 0.0)
    rec  = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if (tp==0 and fp==0 and fn==0) else 0.0)
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else (1.0 if (tp==0 and fp==0 and fn==0) else 0.0)
    return prec, rec, f1

def score_group(df_user, tau=1.0, tol=1, direction="any", require_dir=False):
    g = df_user.sort_values("target_post_idx")
    y_t = g["wb_true"].tolist()
    y_p = g["wb_pred"].tolist()

    true_i, true_s = change_events(y_t, tau=tau, direction=direction)
    pred_i, pred_s = change_events(y_p, tau=tau, direction=direction)

    tp, fp, fn, timing = match_events(pred_i, pred_s, true_i, true_s, tol=tol, require_dir=require_dir)
    prec, rec, f1 = prf1(tp, fp, fn)
    t_mean = float(np.mean(timing)) if len(timing) else math.nan
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "timing_mae_posts": t_mean,
        "n_pred": len(pred_i),
        "n_true": len(true_i),
    }

def main():
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(IN_CSV)
    rows = []

    for tl, g in df.groupby("timeline_id"):
        any_dir = score_group(g, tau=TAU, tol=TOL, direction="any", require_dir=False)
        down    = score_group(g, tau=TAU, tol=TOL, direction="down", require_dir=True)

        rows.append({
            "timeline_id": tl,
            # ANY direction
            "any_tp": any_dir["tp"], "any_fp": any_dir["fp"], "any_fn": any_dir["fn"],
            "any_precision": any_dir["precision"], "any_recall": any_dir["recall"], "any_f1": any_dir["f1"],
            "any_timing_mae_posts": any_dir["timing_mae_posts"],
            "any_true": any_dir["n_true"], "any_pred": any_dir["n_pred"],
            # DOWN only
            "down_tp": down["tp"], "down_fp": down["fp"], "down_fn": down["fn"],
            "down_precision": down["precision"], "down_recall": down["recall"], "down_f1": down["f1"],
            "down_timing_mae_posts": down["timing_mae_posts"],
            "down_true": down["n_true"], "down_pred": down["n_pred"],
        })

    out = pd.DataFrame(rows).sort_values("timeline_id")
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved per-user scores → {OUT_CSV}")

    # Macro (mean over users)
    macro = out[[
        "any_precision","any_recall","any_f1","any_timing_mae_posts",
        "down_precision","down_recall","down_f1","down_timing_mae_posts"
    ]].mean(numeric_only=True)
    print("\nMACRO AVERAGE (mean over users)")
    print(macro)

    # Micro (pool counts then compute)
    any_tp = out["any_tp"].sum(); any_fp = out["any_fp"].sum(); any_fn = out["any_fn"].sum()
    d_tp   = out["down_tp"].sum(); d_fp   = out["down_fp"].sum(); d_fn   = out["down_fn"].sum()
    def pr(tp, fp, fn):
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f = (2*p*r)/(p+r) if (p+r)>0 else 0.0
        return p, r, f
    any_p, any_r, any_f = pr(any_tp, any_fp, any_fn)
    d_p, d_r, d_f = pr(d_tp, d_fp, d_fn)
    print("\nMICRO (pooled counts)")
    print(f"Any  → P {any_p:.3f}  R {any_r:.3f}  F1 {any_f:.3f}  (TP {any_tp}, FP {any_fp}, FN {any_fn})")
    print(f"Down → P {d_p:.3f}    R {d_r:.3f}    F1 {d_f:.3f}    (TP {d_tp}, FP {d_fp}, FN {d_fn})")

if __name__ == "__main__":
    main()
