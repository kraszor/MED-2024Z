import pandas as pd

data = pd.read_csv(
    "results/output_zoo.csv",
    delimiter=",",
)

counts = data.groupby(["cluster_idx", "target"]).size().unstack(fill_value=0)

print(counts)
