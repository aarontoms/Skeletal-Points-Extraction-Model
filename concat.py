import pandas as pd
import glob, os

files = glob.glob("labeled/*.csv")
dfs = [pd.read_csv(f) for f in files]
final = pd.concat(dfs, ignore_index=True)
os.makedirs("combined labeled", exist_ok=True)
final.to_csv("combined labeled/all_labeled.csv", index=False)
print("Saved all_labeled.csv with", len(final), "rows")