def louo_splits(df):
    for tl in df.timeline_id.unique():
        train = df[df.timeline_id != tl].copy()
        test  = df[df.timeline_id == tl].copy()
        yield tl, train, test