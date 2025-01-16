import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import Point


class InputData:
    def __init__(self, file_name: str, sample_size: float):
        self.file_name = file_name
        self.sample_size = sample_size
        self.data = None
        self.target = None
        self.sample_data = None
        self.rest_of_data = None

    def get_sample_data(self) -> None:
        return self.sample_data

    def get_rest_of_data(self) -> None:
        return self.rest_of_data

    def create_transactions(self, data: pd.DataFrame) -> pd.Series:
        return data.apply(
            lambda row: set([f"{k}.{v}" for (k, v) in row.items() if pd.notna(v)]),
            axis=1,
        )

    def create_point_list(
        self, indices: list, index_values: list, transactions: list, targets: list
    ) -> list:
        return [
            Point(init_idx=init_idx, idx=idx, transaction=transaction, target=target)
            for init_idx, idx, transaction, target in zip(
                indices, index_values, transactions, targets
            )
        ]

    def get_data(self) -> None:
        df = pd.read_csv(self.file_name, header=None, delimiter=",")
        self.data = df.iloc[:, 1:]
        self.target = df.iloc[:, 0]

        self.data["transactions"] = self._create_transactions(self.data)

        idx_train, idx_test, y_train, y_test = train_test_split(
            df.index, df.iloc[:, 0].values, train_size=self.sample_size, random_state=42
        )

        self.sample_data = self._create_point_list(
            indices=idx_train.to_numpy(),
            index_values=idx_train.to_frame().reset_index().index.values,
            transactions=self.data.iloc[idx_train.to_list(), 1:].transactions.values,
            targets=y_train,
        )

        self.rest_of_data = self._create_point_list(
            indices=idx_test.to_numpy(),
            index_values=idx_test.to_frame().reset_index().index.values,
            transactions=self.data.iloc[idx_test.to_list(), 1:].transactions.values,
            targets=y_test,
        )
