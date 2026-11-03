import pandas as pd
from sklearn.datasets import fetch_openml
# pyrefly: ignore[missing-import]
from ucimlrepo import fetch_ucirepo

def download_openml(target_id, filename, use_name=False):
    print(f"Fetching {filename}...")
    try:
        if use_name:
            df = fetch_openml(name=target_id, version=1, as_frame=True, parser="auto").frame
        else:
            df = fetch_openml(data_id=target_id, as_frame=True, parser="auto").frame
        df.to_csv(filename, index=False)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def download_uci(target_id, filename):
    print(f"Fetching {filename}...")
    try:
        dataset = fetch_ucirepo(id=target_id)
        df = dataset.data.original
        df.to_csv(filename, index=False)
        print(f"Successfully saved {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

# 1. OpenML Datasets
download_openml("heloc", "FICO_HELOC.csv", use_name=True)
download_openml(42477, "Taiwan_Credit.csv")
download_openml(45577, "Give_Me_Some_Credit.csv")
download_openml(42478, "German_Credit.csv")
download_openml(37, "Pima_Indians_Diabetes.csv")
download_openml(42178, "Telco_Customer_Churn.csv")
download_openml(43696, "HR_Analytics_Attrition.csv")

# 2. UCI Datasets
download_uci(320, "Student_Performance.csv")

print("All dataset downloads attempted.")