import json, glob
import pandas as pd
from pathlib import Path

# candidate names (we'll match them case-insensitively)
CAND_TEXT = ["text", "body", "post"]
CAND_WB   = ["well_being", "well-being", "wellbeing", "wb", "label", "score"]
CAND_AD   = ["adaptive_evidence", "adaptive_spans", "adaptive", "evidence.adaptive"]
CAND_MAL  = ["maladaptive_evidence", "maladaptive_spans", "maladaptive", "evidence.maladaptive"]

def _get_path_ci(d, path):
    """
    Case-insensitive dict lookup with optional dotted paths, e.g. 'evidence.adaptive'.
    Returns None if not found.
    """
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        # map lowercase->original key
        key_map = {k.lower(): k for k in cur.keys()}
        lk = part.lower()
        if lk not in key_map:
            return None
        cur = cur[key_map[lk]]
    return cur

def _first_ci(d, keys, default=None):
    for k in keys:
        v = _get_path_ci(d, k)
        if v is not None:
            return v
    return default

def load_timelines(raw_dir="data/raw"):
    rows = []
    files = sorted(glob.glob(f"{raw_dir}/*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {raw_dir}.")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        tl_id = obj.get("timeline_id") or obj.get("user_id") or Path(fp).stem
        posts = obj.get("posts") or obj.get("data") or []

        for i, p in enumerate(posts):
            text = _first_ci(p, CAND_TEXT, "") or ""
            wb   = _first_ci(p, CAND_WB, None)   # will catch 'Well-being' via case-insensitive match
            ad   = _first_ci(p, CAND_AD, [])
            mal  = _first_ci(p, CAND_MAL, [])

            rows.append({
                "timeline_id": tl_id,
                "post_idx": i,
                "text": text,
                "wb": wb,
                "adaptive": ad if isinstance(ad, list) else [],
                "maladaptive": mal if isinstance(mal, list) else [],
            })

    df = pd.DataFrame(rows).sort_values(["timeline_id", "post_idx"]).reset_index(drop=True)
    # keep only rows with a numeric well-being label (handles strings like "7")
    df = df[pd.to_numeric(df.wb, errors="coerce").notna()].copy()
    df["wb"] = df["wb"].astype(float)
    return df
