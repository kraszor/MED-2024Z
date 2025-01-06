import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyclustering.cluster.rock import rock
from pyclustering.cluster import cluster_visualizer

threshold = 0.8
data = pd.read_csv("my_data.csv", header=None)
features = data.iloc[:, 1:]
encoders = [LabelEncoder() for _ in features.columns]

encoded_features = features.apply(
    lambda col: encoders[features.columns.get_loc(col.name)].fit_transform(col)
)
numeric_data = encoded_features.values.tolist()
# print(numeric_data)
rock_instance = rock(
    data=numeric_data,
    number_clusters=2,
    eps=threshold,
    threshold=threshold,
    ccore=False,
)
rock_instance.process()
clusters = rock_instance.get_clusters()

# print("Clusters:", clusters)
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, numeric_data)
visualizer.show()

"""
UŻYWA ODLEGŁOŚCI EUKLIDESOWSKIEJ WIĘC SIĘ NIE NADAJE DO DANYCH KATEGORYCZNYCH!!!!
"""
