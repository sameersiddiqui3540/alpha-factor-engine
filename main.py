import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_train_data, load_test_data

#Load Train data(first partition parquet file with id=0)
train_data = load_train_data()

# Exploratory Data Analysis (EDA)    
def eda(data):
    print("=======================EDA STARTS=========================")
    # Shape & Schema
    shape = f"\nShape:\n{data.shape}\n"
    print(shape)
    cols = f"Columns:\n{data.columns}\n"
    print(cols)
    schema = f"Schema: \n{data.schema}\n"
    print(schema)
    dtypes = f"Dtypes: \n{data.dtypes}\n"
    print(dtypes)

    # Preview
    first_10_rows = f"First 10 Rows: \n{data.head(10)}\n"
    print(first_10_rows)
    last_10_rows = f"Last 10 Rows: \n{data.tail(10)}\n\n"
    print(last_10_rows)

    # Null/Missing Values
    
    # Null Count Per Column
    null_count = data.select(pl.all().is_null().sum())
    print(f"Null count per Column: \n{null_count}")
    
    # Null Percentage Per Column
    null_pct = data.select(
            (pl.all().is_null().sum()/pl.len() * 100).name.suffix("_null_pct")
        )
    print(f"Null percentage per Column:\n{null_pct}")

    #Statistics
    stats = data.describe()
    print(f"Statistics: \n{stats}")

    #Numeric columns
    num_cols = [col for col, dtype in data.schema.items()
                if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    print(f"Numeric Columns:\n{num_cols}")

    #Value Counts for Categorical/Low-Cardinality Columns
    for col in ["symbol_id", "data_id", "time_id", "responder_6"]:
        if col in data.columns:
            print(f"\f==== {col} ====")
            print(data[col].value_counts().sort("count", descending=True).head())
    
    # Target Variable Distribution
    target_var = "responder_6"
    if target_var in data.columns:
        print(data[target_var].describe())
        print("Skewness:", data[target_var].skew())
        print("Kurtosis:",data[target_var].kurtosis())

        #Quantile Breakdown
        quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        for q in quantiles:
            print(f" q{int(q*100):02d}: {data[target_var].quantile(q):.6f}")
    
    #Correlation Matrix
    corr_mat = data.select(num_cols).to_pandas().corr()
    print(f"Correlation Matrix (Numeric Columns Only): \n{corr_mat}")

    plt.figure(figsize=(20,16))
    sns.heatmap(corr_mat, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    #Feature Distribution (Histograms)
    feature_cols = [c for c in data.columns if c.startswith("feature_")]
    df_pd = data.select(feature_cols[:20]).to_pandas() # First 20 features
    df_pd.hist(bins=50, figsize=(20,15))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()

    #Time-Based Patterns
    #Average target by date
    if "date_id" in data.columns and "responder_6" in data.columns:
        daily_avg = (
            data.group_by("date_id")
            .agg(pl.col("responder_6").mean().alias("avg_target"))
            .sort("date_id")
        )
        print(daily_avg)

    #Average Target by time (Intraday Pattern)
    if "time_id" in data.columns and "responder_6" in data.columns:
        intraday = (
            data.group_by("time_id")
            .agg(pl.col("responder_6").mean().alias("avg_target"))
            .sort("time_id")
            )
        print(intraday)

    #Outlier Detection (Z-score based)
    # find rows where any feature > 5 std devs from the mean
    for col in feature_cols[:10]:
        mean = data[col].mean()
        std = data[col].std()
        if std is not None and std > 0:
            outliers = data.filter(
                ((pl.col(col) - mean) / std).abs() > 5
            )
            if len(outliers) > 0:
                print(f"{col}: {len(outliers)} outliers (>5σ)")

    # Weight Column Check
    if "weight" in data.columns:
        print(data["weight"].describe())
        print("Zero weights:", (data["weight"] == 0).sum())
        print("Null weights:", data["weight"].is_null().sum())

    print("=======================EDA ENDS=========================")

eda(train_data)