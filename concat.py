import pandas as pd
import glob, os
import re

files = glob.glob("labeled/*.csv")
dfs = [pd.read_csv(f) for f in files]
final = pd.concat(dfs, ignore_index=True)
os.makedirs("combined labeled", exist_ok=True)
existing = glob.glob(os.path.join("combined labeled", "all_labeled*.csv"))
nums = []
for p in existing:
    fn = os.path.basename(p)
    m = re.match(r'all_labeled(?:(\d+))?\.csv$', fn)
    if m:
        nums.append(int(m.group(1)) if m.group(1) else 0)

next_num = max(nums) + 1 if nums else 1
out_path = os.path.join("combined labeled", f"all_labeled{next_num}.csv")
final.to_csv(out_path, index=False)
print(f"Saved {out_path} with {len(final)} rows")