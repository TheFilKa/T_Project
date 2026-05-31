from pathlib import Path
import pandas as pd

submission_paths = [
    "data/submissions/submission_rf_ratio_seed_42.csv",
    "data/submissions/submission_rf_ratio_seed_62.csv",
    "data/submissions/submission_rf_ratio_seed_72.csv",
]

subs = [pd.read_csv(path) for path in submission_paths]

out = subs[0][["index"]].copy()
out["IC50"] = sum(s["IC50"] for s in subs) / len(subs)
out["CC50"] = sum(s["CC50"] for s in subs) / len(subs)

# Для ratio-ветки сохраняем химическую логику SI = CC50 / IC50
out["SI"] = out["CC50"] / out["IC50"].clip(lower=1e-9)

output_path = Path("data/submissions/submission_rf_ratio_seedavg_42_62_72.csv")
out.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(out.describe())