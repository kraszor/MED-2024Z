import time
import matplotlib.pyplot as plt
import numpy as np
from main import main


def measure_execution_time(
    file_name, k, theta, approximation_function, drop_outliers, sample_sizes
):
    times = []
    total_data_size = 8124

    for sample_size in sample_sizes:
        actual_data_size = int(sample_size * total_data_size)
        print(f"Testing sample size: {actual_data_size} rows...")

        start_time = time.time()
        main(
            file_name=file_name,
            sample_size=sample_size,
            k=k,
            theta=theta,
            approximation_function=approximation_function,
            drop_outliers=drop_outliers,
        )
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

    return [int(size * total_data_size) for size in sample_sizes], times


file_name = "mashroom"
k = 2
theta = 0.8
approximation_function = lambda x: (1 - x) / (1 + x)
drop_outliers = False

sample_sizes = [0.0125, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75]
x_values, y_values = measure_execution_time(
    file_name=file_name,
    k=k,
    theta=theta,
    approximation_function=approximation_function,
    drop_outliers=drop_outliers,
    sample_sizes=sample_sizes,
)
print(x_values, y_values)
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker="o")
plt.title("Czas działania funkcji w zależności od rozmiaru danych")
plt.xlabel("Liczba danych (liczba wierszy)")
plt.ylabel("Czas działania (sekundy)")
plt.grid()
plt.show()
