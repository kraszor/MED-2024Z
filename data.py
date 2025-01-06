import pandas as pd
from sklearn.model_selection import train_test_split
from utils import Point


class InputData:
    def __init__(self, file_name, sample_size):
        self.file_name = file_name
        self.sample_size = sample_size
        self.data = None
        self.target = None
        self.sample_data = None
        self.rest_of_data = None

    def get_sample_data(self):
        return self.sample_data

    def get_rest_of_data(self):
        return self.rest_of_data

    def get_data(self):
        df = pd.read_csv(self.file_name, header=None, delimiter=",")
        self.data = df.iloc[:, 1:]
        self.target = df.iloc[:, 0]
        self.data["transactions"] = self.data.apply(
            lambda row: set([f"{k}.{v}" for (k, v) in row.items() if pd.notna(v)]),
            axis=1,
        )

        idx_train, idx_test, y_train, y_test = train_test_split(
            df.index,
            df.iloc[:, 0].values,
            train_size=self.sample_size,
            random_state=42,
        )

        self.sample_data = [
            Point(init_idx=init_idx, idx=idx, transaction=transaction, target=y)
            for init_idx, idx, transaction, y in zip(
                idx_train.to_numpy(),
                idx_train.to_frame().reset_index().index.values,
                self.data.iloc[idx_train.to_list(), 1:].transactions.values,
                y_train,
            )
        ]
        self.rest_of_data = [
            Point(init_idx=init_idx, idx=idx, transaction=transaction, target=y)
            for init_idx, idx, transaction, y in zip(
                idx_test.to_numpy(),
                idx_test.to_frame().reset_index().index.values,
                self.data.iloc[idx_test.to_list(), 1:].transactions.values,
                y_test,
            )
        ]
