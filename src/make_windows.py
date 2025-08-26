def make_windows(df_user, k=3, h=1):
    X, y = [], []
    rows = df_user.sort_values("post_idx").to_dict("records")
    for t in range(k-1, len(rows)-h):
        ctx_texts = [rows[t-i]["text"] for i in range(k-1, -1, -1)]
        X.append(" [SEP] ".join(ctx_texts))
        y.append(rows[t+1]["wb"])
    return X, y