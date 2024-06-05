import pandas as pd
import numpy as np
from dask import dataframe as dd
from multiprocessing import Pool
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Функция для параллельной обработки данных
def process_chunk(chunk):
    initial_count = len(chunk)

    # Удаление пустых/NA строк
    non_na_count = chunk.dropna().shape[0]
    na_removed = initial_count - non_na_count
    chunk = chunk.dropna()

    # Удаление дубликатов
    non_duplicates_count = chunk.drop_duplicates().shape[0]
    duplicates_removed = non_na_count - non_duplicates_count
    chunk = chunk.drop_duplicates()

    # Преобразование строк без цифр в пустые
    digit_count_before = (
        chunk["string_no_digits"]
        .apply(lambda x: any(char.isdigit() for char in x))
        .sum()
    )
    chunk["string_no_digits"] = chunk["string_no_digits"].apply(
        lambda x: "" if not any(char.isdigit() for char in x) else x
    )
    digit_count_after = (
        chunk["string_no_digits"]
        .apply(lambda x: any(char.isdigit() for char in x))
        .sum()
    )
    strings_converted = digit_count_before - digit_count_after

    # Удаление записей в промежутке от 1 до 3 часов ночи
    chunk["datetime"] = pd.to_datetime(chunk["datetime"])
    count_before_time_filter = len(chunk)
    chunk = chunk[~((chunk["datetime"].dt.hour >= 1) & (chunk["datetime"].dt.hour < 3))]
    count_after_time_filter = len(chunk)
    time_filtered_out = count_before_time_filter - count_after_time_filter

    # Логирование
    logging.info(f"Обработанный фрагментво: {initial_count} записей")
    logging.info(f"Удалено строк с пропущенными значениями: {na_removed}")
    logging.info(f"Удалено дублирующихся строк: {duplicates_removed}")
    logging.info(f"Строки преобразованы в пустые значения: {strings_converted}")
    logging.info(
        f"Записи удалены в интервале времени с 1 до 3 ночи: {time_filtered_out}"
    )

    return chunk


# Параллельное выполнение обработки
def parallel_processing(input_file, output_file, chunksize=100000):
    # Считывание файла с использованием Dask для обработки больших данных
    ddf = dd.read_csv(input_file, assume_missing=True)

    # Применение обработки к каждому куску данных
    with Pool() as pool:
        results = pool.map(
            process_chunk, [chunk.compute() for chunk in ddf.to_delayed()]
        )

    # Объединение результатов
    processed_df = pd.concat(results)

    # Сохранение обработанного файла
    processed_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = "generated_dataset.csv"
    output_file = "processed_dataset.csv"
    parallel_processing(input_file, output_file)

    logging.info(f"Обработанный набор данных сохранен в {output_file}")
