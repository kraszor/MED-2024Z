from data import InputData
from rock import RockAlgorithm
from utils import save_data_to_csv


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
    # rock.collect_results()
    # df = rock.get_rock_output(my_data)
    # print(df.head())
    # save_data_to_csv(df, f"output_{file_name}")


if __name__ == "__main__":
    file_name = "zoo"
    sample_size = 0.33
    k = 7
    theta = 0.7
    approximation_function = lambda x: (1 + x) / (1 - x)
    drop_outliers = False
    main(
        file_name=file_name,
        sample_size=sample_size,
        k=k,
        theta=theta,
        approximation_function=approximation_function,
        drop_outliers=drop_outliers,
    )
