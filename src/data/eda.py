from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Базовые пути проекта.
# Здесь меняется источник данных и директории для сохранения артефактов.
ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORT_DIR = ROOT / "artifacts" / "reports"
FIGURES_DIR = ROOT / "artifacts" / "figures" / "eda"


def prepare_dataframe(path: Path) -> pd.DataFrame:
    # Базовая подготовка датасета перед любыми проверками:
    # 1. читаем CSV
    # 2. убираем лишние пробелы в строковых колонках
    # 3. отдельно помечаем "пустой" TotalCharges
    # 4. переводим TotalCharges в число для числового анализа
    #
    # Здесь добавляются первичные преобразования, например:
    # - переименование колонок
    # - приведение Yes/No к 1/0
    # - создание новых флагов до EDA
    df = pd.read_csv(path)

    object_cols = df.select_dtypes(include="object").columns
    for column in object_cols:
        df[column] = df[column].astype(str).str.strip()

    df["TotalCharges_missing_raw"] = df["TotalCharges"].eq("")
    df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def outlier_bounds(series: pd.Series) -> tuple[float, float, int]:
    # Вспомогательная функция для поиска выбросов по правилу IQR.
    # Здесь меняется подход к поиску выбросов, например на:
    # - Z-score
    # - percentile clipping
    # - business thresholds
    valid = series.dropna()
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((valid < lower) | (valid > upper)).sum())
    return float(lower), float(upper), count


def build_quality_summary(df: pd.DataFrame) -> dict[str, object]:
    # Главный блок проверок качества данных.
    # Здесь собраны:
    # - дубликаты
    # - пропуски
    # - логические несоответствия между колонками
    # - базовые числовые sanity checks
    #
    # Здесь добавляются пользовательские эвристики и дополнительные проверки.
    # Например:
    # - проверка допустимых категорий
    # - поиск отрицательных значений
    # - контроль диапазонов для бизнес-полей
    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    summary: dict[str, object] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1] - 2),
        "duplicate_rows": int(df.drop(columns=["TotalCharges_missing_raw", "TotalCharges_num"]).duplicated().sum()),
        "duplicate_customer_id": int(df["customerID"].duplicated().sum()),
        "duplicate_rows_excluding_customer_id": int(
            df.drop(columns=["customerID", "TotalCharges_missing_raw", "TotalCharges_num"]).duplicated().sum()
        ),
        "blank_total_charges": int(df["TotalCharges_missing_raw"].sum()),
        "missing_total_charges_after_cast": int(df["TotalCharges_num"].isna().sum()),
        "tenure_zero_and_missing_total_charges": int(
            ((df["tenure"] == 0) & (df["TotalCharges_num"].isna())).sum()
        ),
        "tenure_positive_and_missing_total_charges": int(
            ((df["tenure"] > 0) & (df["TotalCharges_num"].isna())).sum()
        ),
        "phone_logic_mismatches": int(
            ((df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")).sum()
        ),
    }

    for column in service_columns:
        summary[f"{column}_logic_mismatches"] = int(
            ((df["InternetService"] == "No") & (df[column] != "No internet service")).sum()
        )

    for column in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        lower, upper, count = outlier_bounds(df[column])
        summary[f"{column}_iqr_lower"] = lower
        summary[f"{column}_iqr_upper"] = upper
        summary[f"{column}_iqr_outliers"] = count
        summary[f"{column}_p01"] = float(df[column].dropna().quantile(0.01))
        summary[f"{column}_p99"] = float(df[column].dropna().quantile(0.99))

    return summary


def rare_category_summary(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    # Сводка по редким категориям в категориальных признаках.
    # Порог threshold меняется для настройки границы редких категорий.
    rows: list[dict[str, object]] = []
    for column in df.select_dtypes(include="object").columns:
        if column == "customerID":
            continue
        if df[column].nunique(dropna=False) > 20:
            continue
        distribution = df[column].value_counts(normalize=True).sort_values()
        for category, share in distribution.items():
            if share < threshold:
                rows.append(
                    {
                        "feature": column,
                        "category": category,
                        "share": round(float(share), 4),
                    }
                )
    rare_df = pd.DataFrame(rows)
    if rare_df.empty:
        return rare_df
    return rare_df.sort_values(["feature", "share", "category"])


def churn_profiles(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Быстрый профиль таргета по ключевым признакам.
    # Здесь меняется список признаков для группировки churn.
    # Можно добавить gender, Partner, Dependents или engineered features.
    features = ["Contract", "PaymentMethod", "InternetService", "SeniorCitizen", "PaperlessBilling"]
    profiles: dict[str, pd.DataFrame] = {}

    for feature in features:
        profile = (
            df.groupby(feature)["Churn"]
            .value_counts(normalize=True)
            .rename("share")
            .reset_index()
        )
        profile["share"] = profile["share"].round(4)
        profiles[feature] = profile

    return profiles


def save_figures(df: pd.DataFrame) -> None:
    # Блок построения графиков.
    # Здесь добавляются новые графики:
    # 1. ниже добавь новый plt.figure(...)
    # 2. построй нужный график
    # 3. задай русский title/xlabel/ylabel
    # 4. сохрани через plt.savefig(...)
    #
    # Примеры графиков для расширения:
    # - churn по tenure-группам
    # - boxplot MonthlyCharges по Churn
    # - countplot по Contract
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    churn_share = df["Churn"].value_counts(normalize=True).sort_index()
    plt.figure(figsize=(6, 4))
    churn_share.plot(kind="bar", color=["#7aa874", "#d66a5e"])
    plt.title("Баланс классов целевой переменной")
    plt.ylabel("Доля")
    plt.xlabel("Отток")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_balance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    df["MonthlyCharges"].plot(kind="hist", bins=30, color="#4c78a8", edgecolor="white")
    plt.title("Распределение ежемесячных платежей")
    plt.xlabel("Ежемесячный платеж")
    plt.ylabel("Количество клиентов")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_charges_distribution.png", dpi=150)
    plt.close()

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda series: (series == "Yes").mean())
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(7, 4))
    contract_churn.plot(kind="bar", color="#f2a541")
    plt.title("Доля оттока по типу контракта")
    plt.ylabel("Доля оттока")
    plt.xlabel("Тип контракта")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_rate_by_contract.png", dpi=150)
    plt.close()


def write_report(
    summary: dict[str, object],
    profiles: dict[str, pd.DataFrame],
    rare_categories: pd.DataFrame,
) -> Path:
    # Генерация markdown-отчёта.
    # Здесь добавляются новые разделы и интерпретации в итоговый отчёт.
    # Для расширения отчёта обычно:
    # - добавить новый раздел в lines
    # - подставить туда свои метрики или интерпретации
    # - при необходимости сохранить дополнительные таблицы в CSV
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "eda_telco_churn_report.md"

    recommendations = [
        "Привести `TotalCharges` к числовому типу и отдельно обработать 11 пустых значений как новых клиентов с `tenure = 0`.",
        "Сохранить проверку логических связей между сервисами в preprocessing, чтобы ловить грязные загрузки в будущем.",
        "Учесть дисбаланс таргета: доля churn около 26.5%, поэтому на этапе моделирования полезны stratified split и метрики ROC-AUC/PR-AUC/F1.",
        "Проверить влияние редких, но сильных сигналов: `Month-to-month`, `Electronic check`, `SeniorCitizen = 1`.",
        "Для устойчивости модели рассмотреть признаки `avg_monthly_total_ratio` и `is_new_customer`, чтобы аккуратно учитывать клиентов с нулевым стажем.",
    ]

    lines = [
        "# EDA Report: Telco Customer Churn",
        "",
        "## Dataset overview",
        f"- Rows: {summary['rows']}",
        f"- Original feature columns: {summary['columns']}",
        f"- Duplicate full rows: {summary['duplicate_rows']}",
        f"- Duplicate `customerID`: {summary['duplicate_customer_id']}",
        f"- Duplicate rows excluding `customerID`: {summary['duplicate_rows_excluding_customer_id']}",
        "",
        "## Data quality checks",
        f"- Blank `TotalCharges` values: {summary['blank_total_charges']}",
        f"- Missing `TotalCharges` after numeric cast: {summary['missing_total_charges_after_cast']}",
        f"- `tenure = 0` and missing `TotalCharges`: {summary['tenure_zero_and_missing_total_charges']}",
        f"- `tenure > 0` and missing `TotalCharges`: {summary['tenure_positive_and_missing_total_charges']}",
        f"- Phone-service logical mismatches: {summary['phone_logic_mismatches']}",
        f"- Internet-service logical mismatches: {sum(summary[key] for key in summary if key.endswith('_logic_mismatches') and key != 'phone_logic_mismatches')}",
        "",
        "## Numeric sanity checks",
        f"- `tenure` IQR outliers: {summary['tenure_iqr_outliers']} (bounds {summary['tenure_iqr_lower']:.2f} .. {summary['tenure_iqr_upper']:.2f})",
        f"- `MonthlyCharges` IQR outliers: {summary['MonthlyCharges_iqr_outliers']} (bounds {summary['MonthlyCharges_iqr_lower']:.2f} .. {summary['MonthlyCharges_iqr_upper']:.2f})",
        f"- `TotalCharges_num` IQR outliers: {summary['TotalCharges_num_iqr_outliers']} (bounds {summary['TotalCharges_num_iqr_lower']:.2f} .. {summary['TotalCharges_num_iqr_upper']:.2f})",
        f"- `MonthlyCharges` p01/p99: {summary['MonthlyCharges_p01']:.2f} / {summary['MonthlyCharges_p99']:.2f}",
        f"- `TotalCharges_num` p01/p99: {summary['TotalCharges_num_p01']:.2f} / {summary['TotalCharges_num_p99']:.2f}",
        "",
        "## Churn profiles",
    ]

    for feature, profile in profiles.items():
        churn_yes = profile[profile["Churn"] == "Yes"].copy()
        lines.append(f"### {feature}")
        for _, row in churn_yes.iterrows():
            lines.append(f"- {row[feature]}: churn rate {row['share']:.2%}")
        lines.append("")

    if not rare_categories.empty:
        rare_lines = ["| feature | category | share |", "| --- | --- | --- |"]
        for _, row in rare_categories.iterrows():
            rare_lines.append(f"| {row['feature']} | {row['category']} | {row['share']:.4f} |")
        rare_section = "\n".join(rare_lines)
    else:
        rare_section = "No rare categories below threshold."

    lines.extend(
        [
            "## Rare categories (<1%)",
            rare_section,
            "",
            "## Heuristics and recommendations",
        ]
    )
    lines.extend([f"- {item}" for item in recommendations])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    # Точка входа: отсюда запускается весь пайплайн EDA.
    # Здесь подключаются дополнительные шаги пайплайна:
    # - сохранение промежуточных таблиц
    # - экспорт cleaned-версии датасета
    # - вызов дополнительных функций с графиками и эвристиками
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = prepare_dataframe(DATASET_PATH)
    summary = build_quality_summary(df)
    rare_categories = rare_category_summary(df)
    profiles = churn_profiles(df)

    save_figures(df)
    report_path = write_report(summary, profiles, rare_categories)

    print(f"EDA report saved to: {report_path}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("Key checks:")
    print(f"- Duplicate customerID: {summary['duplicate_customer_id']}")
    print(f"- Blank TotalCharges: {summary['blank_total_charges']}")
    print(f"- tenure=0 and missing TotalCharges: {summary['tenure_zero_and_missing_total_charges']}")
    print(f"- MonthlyCharges IQR outliers: {summary['MonthlyCharges_iqr_outliers']}")
    print(f"- TotalCharges IQR outliers: {summary['TotalCharges_num_iqr_outliers']}")


if __name__ == "__main__":
    main()
