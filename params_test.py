from data import InputData
from rock import RockAlgorithm
from utils import save_data_to_csv
import time


def main(
    file_name, sample_size, k, theta, approximation_function, drop_outliers, addon=""
):

    my_data = InputData(file_name=f"{file_name}.csv", sample_size=sample_size)
    my_data.get_data()
    start_time = time.time()
    rock = RockAlgorithm(
        data=my_data,
        k=k,
        theta=theta,
        approximation_function=approximation_function,
        drop_outliers=drop_outliers,
    )

    rock.run()
    diff = time.time() - start_time
    rock.collect_results()
    df = rock.get_rock_output(my_data)
    save_data_to_csv(df, f"output_{file_name}_{k}_{theta}_{addon}")
    print(f"Time: {diff} - {file_name}_{k}_{theta}_{addon}")


if __name__ == "__main__":
    file_name = "voting_records_cleaned"
    sample_size = 0.5
    k = 2
    theta = 0.73
    approximation_function = lambda x: (1 - x) / (1 + x)
    drop_outliers = False
    ks = [2, 4, 6, 8, 10]
    thetas = [0.5, 0.6, 0.7, 0.8]
    approximation_functions = [
        lambda x: (1 - x) / (1 + x),
        lambda x: (1 + x) / (1 - x),
        lambda x: (1 - x),
    ]
    for elem in ks:
        main(
            file_name=file_name,
            sample_size=sample_size,
            k=elem,
            theta=theta,
            approximation_function=approximation_function,
            drop_outliers=drop_outliers,
        )
    for elem in thetas:
        main(
            file_name=file_name,
            sample_size=sample_size,
            k=k,
            theta=elem,
            approximation_function=approximation_function,
            drop_outliers=drop_outliers,
        )
    i = 0
    for elem in approximation_functions:
        i += 1
        main(
            file_name=file_name,
            sample_size=sample_size,
            k=k,
            theta=theta,
            approximation_function=elem,
            drop_outliers=drop_outliers,
            addon=i,
        )
