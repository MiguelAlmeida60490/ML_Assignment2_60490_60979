import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

TRAIN_PATH = "train_data.csv"
TEST_PATH = "test_data.csv"
SAMPLE_PATH = "sample_submission.csv"

ID_COL = "PatientID"     
TARGET_COL = "SurvivalTime"
CENSOR_COL = "Censored"  

OUT_SUBMISSION = "baseline-submission.csv"
PLOT_DIR = "plots_task1"

os.makedirs(PLOT_DIR, exist_ok=True)

# Code from teacher
def error_metric(y, y_hat, c):
    import numpy as np
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]

# Load the 3 .csv files from Kaggle: train, test, sample
def load_data(train_path, test_path, sample_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    print("Data Loaded")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Sample submission shape:", sample.shape)
    return train, test, sample


def standardize_id_names(train_df, test_df, sample_df):
    df_tr = train_df.copy()
    df_te = test_df.copy()
    df_s = sample_df.copy()

    candidates = [ID_COL, "id", "ID", "patient_id", "PatientID", "Unnamed: 0", "index"]
    found = None
    for c in candidates:
        if c in df_tr.columns:
            found = c
            break
    if found is None:
        for c in candidates:
            if c in df_te.columns:
                found = c
                break

    if found is None:
        df_tr[ID_COL] = np.arange(len(df_tr))
        df_te[ID_COL] = np.arange(len(df_te))
        sample_cols = df_s.columns.tolist()
        if sample_cols:
            df_s = df_s.rename(columns={sample_cols[0]: ID_COL})
        print("No ID column found: created 'PatientID' for train & test.")
    else:
        if found != ID_COL:
            if found in df_tr.columns:
                df_tr = df_tr.rename(columns={found: ID_COL})
            if found in df_te.columns:
                df_te = df_te.rename(columns={found: ID_COL})
            sample_cols = df_s.columns.tolist()
            if sample_cols and sample_cols[0] == found:
                df_s = df_s.rename(columns={found: ID_COL})

    return df_tr, df_te, df_s

def dataset_overview(train):
    print("\n=== Dataset overview ===")
    print("Columns:", train.columns.tolist())
    print("Missing per column:")
    print(train.isna().sum())
    if TARGET_COL in train.columns:
        labeled = train[TARGET_COL].notna().sum()
        unlabeled = train[TARGET_COL].isna().sum()
        print(f"Labeled rows (target present): {labeled}")
        print(f"Unlabeled rows (target missing): {unlabeled}")
    if CENSOR_COL in train.columns:
        try:
            cens = int(train[CENSOR_COL].sum())
        except Exception:
            cens = train[CENSOR_COL].value_counts().to_dict()
        print(f"Censoring counts (CENSOR_COL values):\n{train[CENSOR_COL].value_counts(dropna=False)}")
def missingness_plots(df, prefix="train"):
    msno.bar(df, figsize=(16,8))
    plt.title("Missing values - bar")
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_missing_bar.png"), bbox_inches="tight")
    plt.show()

    msno.matrix(df, figsize=(24,12))
    plt.title("Missing values - matrix")
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_missing_matrix.png"), bbox_inches="tight")
    plt.show()

    msno.heatmap(df, figsize=(16,12))
    plt.title("Missingness heatmap")
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_missing_heatmap.png"), bbox_inches="tight")
    plt.show()

    msno.dendrogram(df, figsize=(20,12))
    plt.title("Missingness dendrogram")
    plt.savefig(os.path.join(PLOT_DIR, f"{prefix}_missing_dendrogram.png"), bbox_inches="tight")
    plt.show()

def drop_all_rows_with_any_missing_and_censored(train):
    df = train.copy()
    df = df.dropna(axis=0, how="any")
    if CENSOR_COL in df.columns:
        df = df[df[CENSOR_COL] == 0]
    print("\nStrategy A: after dropping rows with any missing values + censored rows -> rows left:", df.shape[0])
    return df

def baseline_drop_columns_and_censored(train):
    protected = {TARGET_COL, CENSOR_COL}
    cols_with_missing = [c for c in train.columns if train[c].isna().any() and c not in protected]
    df = train.drop(columns=cols_with_missing)
    print("\nDropped columns (those with ANY missing, excluding target & censor):", cols_with_missing)
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL].notna()]
    if CENSOR_COL in df.columns:
        df = df[df[CENSOR_COL] == 0]
    print("Strategy B (assignment baseline) -> remaining rows:", df.shape[0])
    return df, cols_with_missing

def pairplot_numeric_vs_target(df, max_features=6, prefix="pairplot"):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in {ID_COL, TARGET_COL, CENSOR_COL}]
    chosen = numeric[:max_features]
    if not chosen:
        print("No numeric features available for pairplot.")
        return
    cols = chosen + [TARGET_COL]
    sns.pairplot(df[cols], diag_kind="kde", plot_kws={"s": 20, "alpha": 0.6})
    plt.suptitle("Pairplot (selected numeric features vs target)", y=1.02)
    out = os.path.join(PLOT_DIR, f"{prefix}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.show()
    print("Saved pairplot to:", out)

def prepare_train_X_y(df):
    df = df.copy()
    df = df.drop(columns=[ID_COL], errors="ignore")
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    X = X.drop(columns=[CENSOR_COL], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    print("Prepared X shape:", X.shape)
    return X, y

def prepare_test_X(test_df, X_train_columns):
    df = test_df.copy()
    df = df.drop(columns=[ID_COL], errors="ignore")
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=X_train_columns, fill_value=0)
    print("Prepared test X shape:", df.shape)
    return df

def build_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])

def evaluate_kfold(X, y, k=5):
    if len(y) < k:
        print("Not enough examples for k-fold CV.")
        return None, None
    model = build_pipeline()
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
    c = np.zeros_like(y) 
    score = error_metric(y, preds, c)
    print(f"K-fold ({k}) CV error_metric (cMSE): {score:.6f}")
    plot_y_vs_yhat(y, preds, title=f"{k}-fold CV predictions", fname=os.path.join(PLOT_DIR, f"y_vs_yhat_cv_k{k}.png"))
    return score, preds

def evaluate_holdout(X, y, test_size=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    model = build_pipeline()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    c = np.zeros_like(y_te)
    score = error_metric(y_te, preds, c)
    print(f"Hold-out ({int(test_size*100)}%) error_metric (cMSE): {score:.6f}")
    plot_y_vs_yhat(y_te, preds, title="Hold-out predictions", fname=os.path.join(PLOT_DIR, "y_vs_yhat_holdout.png"))
    return score, model, (X_te, y_te, preds)

def plot_y_vs_yhat(y_true, y_pred, title="y vs y_hat", fname=None):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel("True SurvivalTime")
    plt.ylabel("Predicted SurvivalTime")
    plt.title(title)
    plt.grid(True)
    if fname:
        plt.savefig(fname, bbox_inches="tight")
        print("Saved plot:", fname)
    plt.show()

def generate_submission(final_model, X_test_df, sample_submission, filename=OUT_SUBMISSION, clip_nonneg=False):
    preds = final_model.predict(X_test_df)
    if clip_nonneg:
        preds = np.clip(preds, 0, None)
    sub = sample_submission.copy()
    cols = sub.columns.tolist()
    if len(cols) >= 2:
        id_col_name = cols[0]
        target_col_name = cols[1]
        sub[target_col_name] = preds
        sub = sub.rename(columns={id_col_name: ID_COL, target_col_name: TARGET_COL})
    else:
        if ID_COL in sub.columns:
            ids = sub[ID_COL].values
        else:
            ids = np.arange(len(preds))
        sub = pd.DataFrame({ID_COL: ids, TARGET_COL: preds})
    sub.to_csv(filename, index=False)
    print("Saved submission to:", filename)

def main():
    train_df, test_df, sample_sub = load_data(TRAIN_PATH, TEST_PATH, SAMPLE_PATH)
    train_df, test_df, sample_sub = standardize_id_names(train_df, test_df, sample_sub)

    dataset_overview(train_df)
    missingness_plots(train_df, prefix="train")

    df_allrows = drop_all_rows_with_any_missing_and_censored(train_df)

    df_baseline, dropped_cols = baseline_drop_columns_and_censored(train_df)

    if "Unnamed: 0" in df_baseline.columns:
        df_baseline = df_baseline.drop(columns=["Unnamed: 0"])
        print("Dropped 'Unnamed: 0' then.")

    pairplot_numeric_vs_target(df_baseline, max_features=6, prefix="pairplot_baseline")

    X, y = prepare_train_X_y(df_baseline)
    print("\nBaseline training rows (after filtering):", X.shape[0], "features:", X.shape[1])

    cv_score, _ = evaluate_kfold(X, y, k=5)

    hold_score, _, _ = evaluate_holdout(X, y, test_size=0.2)

    final_model = build_pipeline()
    final_model.fit(X, y)
    train_preds = final_model.predict(X)
    train_c = np.zeros_like(y)
    train_score = error_metric(y, train_preds, train_c)
    print("Final model train error_metric (cMSE) on baseline data:", train_score)

    X_test_prepared = prepare_test_X(test_df, X.columns)

    generate_submission(final_model, X_test_prepared, sample_sub, filename=OUT_SUBMISSION, clip_nonneg=False)

    print("\n--- Summary ---")
    print("Original train rows:", len(train_df))
    print("Dropped columns (assignment baseline):", dropped_cols)
    print("Rows if drop all rows with any missing + censored:", df_allrows.shape[0])
    print("Rows after assignment baseline filtering (uncensored + target present):", X.shape[0])
    print("K-fold CV error_metric:", cv_score)
    print("Hold-out error_metric:", hold_score)
    print("Train error_metric (final model on full baseline data):", train_score)
    
if __name__ == "__main__":
    main()