import pandas as pd


def inspect_dataframe(df: pd.DataFrame) -> None:
    """
    Print a comprehensive inspection report for any pandas DataFrame.

    Covers shape, data types, memory usage, missing values, duplicate rows,
    numeric and categorical descriptive statistics, unique value counts per
    categorical column, and a head and tail sample.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    """
    sep = "-" * 60

    print("SHAPE AND STRUCTURE")
    print(sep)
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print()

    print("COLUMN NAMES AND DATA TYPES")
    print(sep)
    print(df.dtypes.to_string())
    print()

    print("MEMORY USAGE")
    print(sep)
    total_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Total: {total_mb:.2f} MB")
    print()

    print("MISSING VALUES")
    print(sep)
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    missing = pd.DataFrame({"Null Count": null_counts, "Null %": null_pct})
    missing = missing[missing["Null Count"] > 0].sort_values("Null Count", ascending=False)
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())
    print()

    print("DUPLICATE ROWS")
    print(sep)
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print()

    print("DESCRIPTIVE STATISTICS (NUMERIC)")
    print(sep)
    print(df.describe().to_string())
    print()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(categorical_cols) > 0:
        print("DESCRIPTIVE STATISTICS (CATEGORICAL)")
        print(sep)
        print(df[categorical_cols].describe().to_string())
        print()

        print("UNIQUE VALUES PER CATEGORICAL COLUMN")
        print(sep)
        for col in categorical_cols:
            print(f"{col}: {df[col].nunique()} unique values")
        print()

    print("SAMPLE DATA")
    print(sep)
    print("First 5 rows:")
    print(df.head().to_string())
    print()
    print("Last 5 rows:")
    print(df.tail().to_string())
