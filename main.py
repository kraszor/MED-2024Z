from data import InputData
from rock import RockAlgorithm
from utils import save_data_to_csv
import time


def main(file_name, sample_size, k, theta, approximation_function, drop_outliers):

    my_data = InputData(file_name=f"{file_name}.csv", sample_size=sample_size)
    my_data.get_data()

    rock = RockAlgorithm(
        data=my_data,
        k=k,
        theta=theta,
        approximation_function=approximation_function,
        drop_outliers=drop_outliers,
    )

    rock.run()


if __name__ == "__main__":
    file_name = "voting_records_cleaned"
    sample_size = 0.5
    k = 2
    theta = 0.73
    approximation_function = lambda x: (1 + x) / (1 - x)
    drop_outliers = False
    start = time.time()
    main(
        file_name=file_name,
        sample_size=sample_size,
        k=k,
        theta=theta,
        approximation_function=approximation_function,
        drop_outliers=drop_outliers,
    )
    print("TIME: ", time.time() - start)
