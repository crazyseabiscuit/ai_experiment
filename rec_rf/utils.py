import pandas as pd
from collections import defaultdict
import csv
def partition_items(df):
    """
    return a partitioned map, where each cluster is a set sampled
    from its "nearest" items
    """
    k = df.shape[0]
    n_items = df.shape[1]

    clusters = defaultdict(set)
    selected = set()
    i = 0

    while len(selected) < n_items:
        label = i % k
        i += 1

        row = df.loc[label, :].values.tolist()

        gradient = {item: dist for item, dist in enumerate(row) if item not in selected}

        nearest_item = min(gradient, key=gradient.get)
        selected.add(nearest_item)
        clusters[label].add(nearest_item)

    return clusters

def get_scaled_clusters(centers):
    """
    return a DataFrame with the item-to-center "distance" scaled
    within `[0.0, 1.0]` where `0.0` represents "nearest"
    """
    df_scaled = pd.DataFrame()

    df = pd.DataFrame(centers)
    n_items = df.shape[1]

    for item in range(n_items):
        row = df[item].values
        item_max = max(row)
        item_min = min(row)
        scale = item_max - item_min

        df_scaled[item] = pd.Series([1.0 - (val - item_min) / scale for val in row])

    return df_scaled


NO_RATING = "99"
MAX_RATING = 10.0
def load_data(data_path):
    rows = []

    with open(data_path, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")

        for row in csvreader:
            conv = [None] * (len(row) - 1)

            for i in range(1, len(row)):
                if row[i] != NO_RATING:
                    rating = float(row[i]) / MAX_RATING
                    conv[i - 1] = rating

            rows.append(conv)

    return rows