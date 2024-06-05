import pandas as pd
import numpy as np
from faker import Faker
import random

# Инициализация Faker
fake = Faker()

# Параметры датасета
num_records = (
    10000000  # Количество записей, при необходимости уменьшить до разумного размера
)
duplicate_percentage = 0.1  # Процент дубликатов
num_duplicates = int(num_records * duplicate_percentage)

# Генерация данных
data = {
    "numeric": [random.randint(1, 100) for _ in range(num_records)],
    "datetime": [
        fake.date_time_between(start_date="-5y", end_date="now")
        for _ in range(num_records)
    ],
    "string_no_digits": [fake.word() for _ in range(num_records)],
    "empty_or_na": [
        None if random.random() < 0.1 else fake.word() for _ in range(num_records)
    ],
}

# Создание DataFrame
df = pd.DataFrame(data)

# Добавление дубликатов
duplicates = df.sample(num_duplicates)
df = pd.concat([df, duplicates], ignore_index=True)

# Перемешивание строк
df = df.sample(frac=1).reset_index(drop=True)

# Сохранение в CSV
output_file = "generated_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Датасет сохранен в {output_file}")
