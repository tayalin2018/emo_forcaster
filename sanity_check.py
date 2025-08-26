import glob, json, os
print("CWD:", os.getcwd())

files = sorted(glob.glob("data/raw/*.json"))
print("#json in data/raw:", len(files))
if not files:
    raise SystemExit("No .json files in data/raw")

fp = files[0]
print("Example file:", fp)
with open(fp, "r", encoding="utf-8") as f:
    obj = json.load(f)

if isinstance(obj, dict):
    posts = obj.get("posts") or obj.get("data") or []
    print("Top-level keys:", list(obj.keys()))
else:
    posts = obj
    print("Top-level is a list of posts")

print("First post keys:", list(posts[0].keys()))
print("First post sample well-being field values:",
      {k: posts[0].get(k, None) for k in ["well_being","well-being","wellbeing","wb","label","score"]})
