import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt

IN_CSV  = "reports/gru_preds.csv"
OUT_DIR = "reports/plots"
TAU = 1.0  # change-point threshold in well-being points

def change_points(arr, tau=1.0):
    arr = np.array(arr, dtype=float)
    d = np.diff(arr)
    # return positions (index of the point AFTER the jump)
    return np.where(np.abs(d) >= tau)[0] + 1

def plot_user(df_user, out_path, tau=1.0):
    # sort by post index for consistent x-axis
    df_user = df_user.sort_values("target_post_idx")
    x   = df_user["target_post_idx"].tolist()
    y_t = df_user["wb_true"].tolist()
    y_p = df_user["wb_pred"].tolist()

    cps_t = change_points(y_t, tau=tau)
    cps_p = change_points(y_p, tau=tau)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y_t, marker="o", linewidth=2, label="True well-being")
    plt.plot(x, y_p, marker="s", linestyle="--", linewidth=2, label="Predicted")

    # vertical markers for change-points
    for c in cps_t:
        plt.axvline(x=x[c], linestyle=":", linewidth=1)
    for c in cps_p:
        # small offset so predicted vlines don't perfectly overlap
        plt.axvline(x=x[c] + 0.1, linestyle="--", linewidth=1)

    plt.xlabel("Target post index")
    plt.ylabel("Well-being (1–10)")
    plt.ylim(0.5, 10.5)
    plt.title(f"Timeline: {df_user['timeline_id'].iloc[0]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    for tl, g in df.groupby("timeline_id"):
        out = os.path.join(OUT_DIR, f"{tl}.png")
        plot_user(g, out, tau=TAU)
        print("Saved", out)

if __name__ == "__main__":
    main()
