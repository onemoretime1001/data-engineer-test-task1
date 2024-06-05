import pandas as pd
import scipy.stats as stats

# Данные
data = {
    "company": ["competitor", "our_company"],
    "successful_prototypes": [5, 0],
    "total_prototypes": [1000, 201],
}

# Создаем DataFrame
df = pd.DataFrame(data)


# 1. Частотный метод
def frequentist_method(df):
    total_success = df["successful_prototypes"].sum()
    total_prototypes = df["total_prototypes"].sum()
    success_probability = total_success / total_prototypes
    return success_probability


# 2. Байесовский метод
def bayesian_method(df):
    # Параметры априорного Beta распределения
    alpha_prior = 1
    beta_prior = 1

    # Обновляем параметры на основе наших данных
    alpha_posterior = (
        alpha_prior
        + df.loc[df["company"] == "our_company", "successful_prototypes"].values[0]
    )
    beta_posterior = (
        beta_prior
        + df.loc[df["company"] == "our_company", "total_prototypes"].values[0]
        - df.loc[df["company"] == "our_company", "successful_prototypes"].values[0]
    )

    # Вычисление вероятности успеха следующего прототипа
    mean_posterior = stats.beta.mean(alpha_posterior, beta_posterior)
    return mean_posterior


# Рассчитаем вероятность успеха для обоих методов
frequentist_probability = frequentist_method(df)
bayesian_probability = bayesian_method(df)

print(
    f"Вероятность успеха следующего прототипа по частотному методу: {frequentist_probability:.4f}"
)
print(
    f"Вероятность успеха следующего прототипа по Байесовскому методу: {bayesian_probability:.4f}"
)
