import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


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

def error_metric(y, y_hat, c):
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    c = np.asarray(c).astype(int)
    err = y - y_hat
    err = (1 - c) * (err ** 2) + c * (np.maximum(0, err) ** 2)
    return np.sum(err) / err.shape[0]

def find_id_col(df):
    candidates = [ID_COL, "id", "ID", "patient_id", "PatientID", "Unnamed: 0", "index"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_data(train_path, test_path, sample_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    print("Loaded: train", train.shape, "test", test.shape, "sample", sample.shape)
    return train, test, sample

def standardize_id_names(train_df, test_df, sample_df):
    df_tr = train_df
    df_te = test_df
    df_s = sample_df

    found = find_id_col(df_tr) or find_id_col(df_te) or find_id_col(df_s)
    if found is None:
        df_tr = df_tr.copy()
        df_te = df_te.copy()
        df_tr[ID_COL] = np.arange(len(df_tr))
        df_te[ID_COL] = np.arange(len(df_te))
        if df_s.shape[1] >= 1:
            df_s = df_s.copy()
            df_s = df_s.rename(columns={df_s.columns[0]: ID_COL})
        print("No ID found: created PatientID for train & test.")
        return df_tr, df_te, df_s

    if found != ID_COL:
        if found in df_tr.columns:
            df_tr = df_tr.rename(columns={found: ID_COL})
        if found in df_te.columns:
            df_te = df_te.rename(columns={found: ID_COL})
        if found in df_s.columns:
            df_s = df_s.rename(columns={found: ID_COL})
    return df_tr, df_te, df_s

def dataset_overview(train_df):
    print("\n=== Dataset overview ===")
    print("Columns:", train_df.columns.tolist())
    print("Missing per column:\n", train_df.isna().sum())
    if TARGET_COL in train_df.columns:
        print("Target present:", train_df[TARGET_COL].notna().sum(), "missing:", train_df[TARGET_COL].isna().sum())
    if CENSOR_COL in train_df.columns:
        print("Censor distribution:\n", train_df[CENSOR_COL].value_counts(dropna=False))




def pairplot_numeric_vs_target(df, max_features=6, prefix="pairplot"):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in {ID_COL, TARGET_COL, CENSOR_COL}]
    chosen = numeric[:max_features]
    if not chosen or TARGET_COL not in df.columns:
        print("Not enough numeric features or no target for pairplot.")
        return
    cols = chosen + [TARGET_COL]
    sns.pairplot(df[cols].dropna(subset=[TARGET_COL]), diag_kind="kde", plot_kws={"s":20,"alpha":0.6})
    out = os.path.join(PLOT_DIR, f"{prefix}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved pairplot to:", out)




def plot_y_vs_yhat(y_true, y_pred, title="y vs y_hat", fname=None):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel("True SurvivalTime"); plt.ylabel("Predicted SurvivalTime"); plt.title(title)
    if fname:
        plt.savefig(fname, bbox_inches="tight")
        print("Saved plot:", fname)
    plt.close()

def prepare_train_X_y(df):
    df_local = df.copy()
    if ID_COL in df_local.columns:
        df_local = df_local.drop(columns=[ID_COL])
    y = df_local[TARGET_COL].values
    X = df_local.drop(columns=[TARGET_COL], errors="ignore")
    X = X.drop(columns=[CENSOR_COL], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    print("Prepared X shape:", X.shape)
    return X, y

def prepare_test_X(test_df, X_train_columns):
    df_local = test_df.copy()
    if ID_COL in df_local.columns:
        df_local = df_local.drop(columns=[ID_COL])
    X_test = pd.get_dummies(df_local, drop_first=True)
    X_test = X_test.reindex(columns=X_train_columns, fill_value=0)
    print("Prepared test X shape:", X_test.shape)
    return X_test


def build_pipeline():
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=15)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

def evaluate_kfold(X, y, k=5):
    if len(y) < k:
        print("Not enough examples for k-fold CV.")
        return None, None
    model = build_pipeline()
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
    c = np.zeros_like(y)
    score = error_metric(y, preds, c)
    print(f"K-fold ({k}) CV cMSE: {score:.6f}")
    plot_y_vs_yhat(y, preds, title=f"{k}-fold CV predictions", fname=os.path.join(PLOT_DIR, f"y_vs_yhat_cv_k{k}.png"))
    return score, preds

def evaluate_holdout(X, y, test_size=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    model = build_pipeline()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    c = np.zeros_like(y_te)
    score = error_metric(y_te, preds, c)
    print(f"Hold-out ({int(test_size*100)}%) cMSE: {score:.6f}")
    plot_y_vs_yhat(y_te, preds, title="Hold-out predictions", fname=os.path.join(PLOT_DIR, "y_vs_yhat_holdout.png"))
    return score, model, (X_te, y_te, preds)

def generate_submission(final_model, X_test_df, sample_submission, filename=OUT_SUBMISSION, clip_nonneg=False):
    preds = final_model.predict(X_test_df)
    if clip_nonneg:
        preds = np.clip(preds, 0, None)
    sub = sample_submission.copy()
    cols = sub.columns.tolist()
    if len(cols) >= 2:
        sub[cols[1]] = preds
    else:
        ids = sub[ID_COL].values if ID_COL in sub.columns else np.arange(len(preds))
        sub = pd.DataFrame({ID_COL: ids, TARGET_COL: preds})
    sub.to_csv(filename, index=False)
    print("Saved submission to:", filename)
    
def get_next_submission_filename(task_num, base_name):
    folder = f"submissions_task_{task_num}"
    os.makedirs(folder, exist_ok=True)

    existing = []
    for f in os.listdir(folder):
        if f.startswith(base_name) and f.endswith(".csv"):
            try:
                num = int(f.replace(base_name + "-", "").replace(".csv", ""))
                existing.append(num)
            except:
                pass

    next_num = 1 if not existing else max(existing) + 1
    next_num_str = f"{next_num:02d}"

    filename = f"{base_name}-{next_num_str}.csv"
    full_path = os.path.join(folder, filename)

    return full_path

def main():
    train_df, test_df, sample_sub = load_data(TRAIN_PATH, TEST_PATH, SAMPLE_PATH)
    train_df, test_df, sample_sub = standardize_id_names(train_df, test_df, sample_sub)

    dataset_overview(train_df)

    # Keep all rows, remove only censored ones
    df_impute = train_df.copy()
    if CENSOR_COL in df_impute.columns:
        df_impute = df_impute[df_impute[CENSOR_COL] == 0]
    
    df_impute = df_impute.dropna(subset=[TARGET_COL])


    X, y = prepare_train_X_y(df_impute)
    print("Training rows:", X.shape[0], "features:", X.shape[1])

    cv_score, _ = evaluate_kfold(X, y, k=5)
    hold_score, _, _ = evaluate_holdout(X, y, test_size=0.2)

    final_model = build_pipeline()
    final_model.fit(X, y)

    train_preds = final_model.predict(X)
    train_c = np.zeros_like(y)
    train_score = error_metric(y, train_preds, train_c)

    print("Final train cMSE:", train_score)

    X_test_prepared = prepare_test_X(test_df, X.columns)
    submission_path = get_next_submission_filename(
        task_num=3,
        base_name="handle-missing-submission-"
    )

    generate_submission(
        final_model,
        X_test_prepared,
        sample_sub,
        filename=submission_path,
        clip_nonneg=False
    )

    print("\n--- Summary ---")
    print("Original train rows:", len(train_df))
    print("Train rows after removing censored:", X.shape[0])
    print("K-fold CV cMSE:", cv_score)
    print("Hold-out cMSE:", hold_score)
    print("Train cMSE:", train_score)


if __name__ == "__main__":
    main()