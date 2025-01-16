import pandas as pd

moja_lista = [
    "output_zoo"
    # "7_0.5_",
    # "7_0.6_",
    # "7_0.7_",
    # "7_0.8_",
    # "2_0.7_",
    # "4_0.7_",
    # "6_0.7_",
    # "8_0.7_",
    # "10_0.7_",
    # "7_0.7_1",
    # "7_0.7_2",
    # "7_0.7_3",
]
for elem in moja_lista:
    print("-" * 10)
    data = pd.read_csv(
        f"results/{elem}.csv",
        delimiter=",",
    )

    counts = data.groupby(["cluster_idx", "target"]).size().unstack(fill_value=0)
    # df = counts.set_index("target").T
    dominant_counts = counts.max(axis=1)
    total_dominant = dominant_counts.sum()
    total_samples = counts.sum().sum()  # Całkowita liczba próbek
    purity = total_dominant / total_samples

    # print(counts.info())
    print(elem)
    print(purity)
    print(counts)
