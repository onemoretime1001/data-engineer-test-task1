import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Считывание обработанного датасета
input_file = "processed_dataset.csv"
df = pd.read_csv(input_file)

# Агрегация по времени (по часам)
df["datetime"] = pd.to_datetime(df["datetime"])
df["hour"] = df["datetime"].dt.hour

aggregated = (
    df.groupby("hour")
    .agg(
        unique_string_count=("string_no_digits", lambda x: x.nunique()),
        numeric_mean=("numeric", "mean"),
        numeric_median=("numeric", "median"),
    )
    .reset_index()
)

print(aggregated.head())

# SQL запрос для выполнения подобных расчетов напрямую в базе данных:
# SELECT
#   EXTRACT(HOUR FROM datetime) AS hour,
#   COUNT(DISTINCT string_no_digits) AS unique_string_count,
#   AVG(numeric) AS numeric_mean,
#   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY numeric) AS numeric_median
# FROM
#   processed_dataset
# GROUP BY
#   EXTRACT(HOUR FROM datetime);

# Мерж исходного датасета с рассчитанными метриками по ближайшему часу
df = df.merge(aggregated, on="hour", how="left")

# Аналитические метрики

# Гистограмма для колонки numeric
plt.figure(figsize=(10, 6))
sns.histplot(df["numeric"], bins=30, kde=True)
plt.title("Гистограмма для колонки numeric")
plt.xlabel("Numeric")
plt.ylabel("Частота")
plt.show()

# Расчет 95% доверительного интервала
numeric_mean = df["numeric"].mean()
numeric_std = df["numeric"].std()
confidence_interval = stats.norm.interval(
    0.95, loc=numeric_mean, scale=numeric_std / np.sqrt(len(df))
)
print(f"95% Confidence Interval: {confidence_interval}")

# Визуализация среднего значения numeric по месяцам
df["month"] = df["datetime"].dt.to_period("M")
monthly_mean = df.groupby("month")["numeric"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x="month", y="numeric", data=monthly_mean)
plt.title("Monthly Mean of Numeric Column")
plt.xlabel("Month")
plt.ylabel("Mean Numeric")
plt.xticks(rotation=45)
plt.show()

# Heatmap по частотности символов в колонке string_no_digits
char_freq = df["string_no_digits"].str.cat(sep="").lower()
char_series = pd.Series(list(char_freq))
char_counts = char_series.value_counts().to_frame().reset_index()
char_counts.columns = ["character", "count"]

plt.figure(figsize=(12, 8))
sns.heatmap(
    char_counts.pivot_table(
        values="count", index="character", columns="character", fill_value=0
    ),
    annot=True,
    fmt="d",
    cmap="viridis",
)
plt.title("Heatmap of Character Frequency in String_no_digits")
plt.show()

# Доп. задание: Случайное разделение датасета на 3 части
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
part1 = df_shuffled[: int(len(df_shuffled) * 0.25)]
part2 = df_shuffled[int(len(df_shuffled) * 0.25) : int(len(df_shuffled) * 0.50)]
part3 = df_shuffled[int(len(df_shuffled) * 0.50) :]

# Проверка на статистическую значимость различий для среднего по колонке numeric
t_stat, p_value = stats.ttest_ind(part1["numeric"], part2["numeric"])
print(f"T-test between part1 and part2: t-stat={t_stat}, p-value={p_value}")

# Оценка силы эффекта (Cohen's d)
mean1, mean2 = part1["numeric"].mean(), part2["numeric"].mean()
std1, std2 = part1["numeric"].std(), part2["numeric"].std()
n1, n2 = len(part1), len(part2)
pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
cohen_d = (mean1 - mean2) / pooled_std
print(f"Cohen's d: {cohen_d}")

# Баесовский подход
import pymc3 as pm

with pm.Model() as model:
    mu1 = pm.Normal("mu1", mu=0, sigma=10)
    mu2 = pm.Normal("mu2", mu=0, sigma=10)
    sigma1 = pm.HalfNormal("sigma1", sigma=10)
    sigma2 = pm.HalfNormal("sigma2", sigma=10)

    obs1 = pm.Normal("obs1", mu=mu1, sigma=sigma1, observed=part1["numeric"])
    obs2 = pm.Normal("obs2", mu=mu2, sigma=sigma2, observed=part2["numeric"])

    diff_of_means = pm.Deterministic("diff_of_means", mu1 - mu2)
    trace = pm.sample(2000, tune=1000)

pm.plot_posterior(trace, var_names=["diff_of_means"])
plt.show()

# Комментарий по выбору методик расчета:
# - Для расчета доверительного интервала использовали нормальное распределение, так как это стандартный метод для больших выборок.
# - Для оценки силы эффекта применили показатель Cohen's d, так как он позволяет оценить размер эффекта при сравнении двух групп.
# - Баесовский подход применили для дополнительной проверки различий между группами, что позволяет получить вероятностное распределение различий.

print("Метрики рассчитаны и визуализации построены.")
