import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.manifold import Isomap
from sklearn.base import BaseEstimator

TRAIN_PATH = "train_data.csv"
TEST_PATH = "test_data.csv"
SAMPLE_PATH = "sample_submission.csv"

ID_COL = "PatientID"
TARGET_COL = "SurvivalTime"
CENSOR_COL = "Censored"

PLOT_DIR = "plots_task4"
SUMMARY_CSV = "task4_cv_summary.csv"
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

KNN_K = 15

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
    print("Loaded shapes:", train.shape, test.shape, sample.shape)
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


def get_next_submission_filename(task_num, base_name):
    folder = f"submissions_task_{task_num}"
    os.makedirs(folder, exist_ok=True)

    existing = []
    for f in os.listdir(folder):
        if f.startswith(base_name) and f.endswith(".csv"):
            try:
                num = int(f.replace(base_name + "-", "").replace(".csv", ""))
                existing.append(num)
            except Exception:
                pass

    next_num = 1 if not existing else max(existing) + 1
    next_num_str = f"{next_num:02d}"

    filename = f"{base_name}-{next_num_str}.csv"
    full_path = os.path.join(folder, filename)
    return full_path

def preprocess_features(train_df, test_df):

    t = train_df.copy()
    s = test_df.copy()

    if ID_COL not in t.columns:
        t[ID_COL] = np.arange(len(t))
    if ID_COL not in s.columns:
        s = s.copy()
        s[ID_COL] = np.arange(len(s))

    excluded = {TARGET_COL, CENSOR_COL}
    feature_cols = [c for c in t.columns if c not in excluded and c != ID_COL]

    num_cols = t[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    if num_cols:
        #num_imp = SimpleImputer(strategy="mean")
        num_imp = KNNImputer(n_neighbors=KNN_K)
        t_num = pd.DataFrame(num_imp.fit_transform(t[num_cols]), columns=num_cols, index=t.index)
        s_num = pd.DataFrame(num_imp.transform(s[num_cols]), columns=num_cols, index=s.index)
    else:
        t_num = pd.DataFrame(index=t.index)
        s_num = pd.DataFrame(index=s.index)

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        t_cat_imp = pd.DataFrame(cat_imp.fit_transform(t[cat_cols]), columns=cat_cols, index=t.index)
        s_cat_imp = pd.DataFrame(cat_imp.transform(s[cat_cols]), columns=cat_cols, index=s.index)
        t_cat = pd.get_dummies(t_cat_imp, drop_first=True)
        s_cat = pd.get_dummies(s_cat_imp, drop_first=True)
    else:
        t_cat = pd.DataFrame(index=t.index)
        s_cat = pd.DataFrame(index=s.index)

    X_train = pd.concat([t_num, t_cat], axis=1)
    X_test = pd.concat([s_num, s_cat], axis=1)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train = X_train.reset_index(drop=True)
    X_train.insert(0, ID_COL, t[ID_COL].reset_index(drop=True))
    X_test = X_test.reset_index(drop=True)
    X_test.insert(0, ID_COL, s[ID_COL].reset_index(drop=True))

    print("Preprocessing features done. Shapes:", X_train.shape, X_test.shape)
    return X_train, X_test, X_train.columns.tolist()


def prepare_X_y_c(train_df, X_train_df):
    if ID_COL not in X_train_df.columns:
        raise ValueError("X_train_df must include ID_COL as a column for safe alignment.")

    merge_cols = [ID_COL]
    if TARGET_COL in train_df.columns:
        merge_cols.append(TARGET_COL)
    if CENSOR_COL in train_df.columns:
        merge_cols.append(CENSOR_COL)

    train_small = train_df[[c for c in merge_cols if c in train_df.columns]].copy()

    merged = pd.merge(X_train_df, train_small, on=ID_COL, how="left", validate="one_to_one")
    X = merged.drop(columns=[ID_COL])
    if TARGET_COL in X.columns:
        X = X.drop(columns=[TARGET_COL])
    if CENSOR_COL in X.columns:
        X = X.drop(columns=[CENSOR_COL])

    y = merged[TARGET_COL].values if TARGET_COL in merged.columns else np.array([np.nan] * len(merged))
    c = merged[CENSOR_COL].fillna(0).astype(int).values if CENSOR_COL in merged.columns else np.zeros(len(merged), dtype=int)

    labeled_mask = ~pd.isna(y)
    unlabeled_mask = pd.isna(y)

    print("prepare_X_y_c: n_rows =", len(X),
          "labeled =", labeled_mask.sum(),
          "unlabeled =", unlabeled_mask.sum())
    return X.reset_index(drop=True), y, c, labeled_mask, unlabeled_mask

def plot_y_vs_yhat(y_true, y_pred, out_path, title="y vs y_hat"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel("True SurvivalTime")
    plt.ylabel("Predicted SurvivalTime")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved plot:", out_path)

class FrozenTransformer(BaseEstimator):
    """
    Wrapper for a pre-fitted transformer (Isomap).
    Provided in the assignment statement.
    """

    def __init__(self, fitted_transformer):
        self.fitted_transformer = fitted_transformer

    def __getattr__(self, name):
        return getattr(self.fitted_transformer, name)

    def __sklearn_clone__(self):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.fitted_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fitted_transformer.transform(X)


def evaluate_model_with_kfold(model, X, y, c, labeled_mask, cv_splits=5, model_name="model"):
    
    labeled_indices = np.where(labeled_mask)[0]
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_errors = []

    oof_preds = np.zeros(labeled_indices.shape[0], dtype=float)

    for fold_idx, (tr_loc, te_loc) in enumerate(kf.split(labeled_indices)):
        tr = labeled_indices[tr_loc]
        te = labeled_indices[te_loc]

        X_tr = X.iloc[tr]
        X_te = X.iloc[te]
        y_tr = y[tr]
        y_te = y[te]
        c_tr = c[tr]
        c_te = c[te]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        err = error_metric(y_te, preds, c_te)
        fold_errors.append(err)
        oof_preds[te_loc] = preds

        print(f"[{model_name}] Fold {fold_idx+1}/{cv_splits} cMSE={err:.6f}")

    mean_err = float(np.mean(fold_errors))
    std_err = float(np.std(fold_errors))
    print(f"[{model_name}] mean cMSE={mean_err:.6f}, std={std_err:.6f}")

    return fold_errors, oof_preds, mean_err, std_err

def main():

    train_df, test_df, sample_sub = load_data(TRAIN_PATH, TEST_PATH, SAMPLE_PATH)
    train_df, test_df, sample_sub = standardize_id_names(train_df, test_df, sample_sub)

    X_train_pre, X_test_pre, feature_cols = preprocess_features(train_df, test_df)

    X_all, y_all, c_all, labeled_mask, unlabeled_mask = prepare_X_y_c(train_df, X_train_pre)

    X_labeled = X_all[labeled_mask]
    y_labeled = y_all[labeled_mask]
    c_labeled = c_all[labeled_mask]

    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    base_folds, base_oof, base_mean, base_std = evaluate_model_with_kfold(
        baseline_pipe, X_all, y_all, c_all, labeled_mask,
        cv_splits=5,
        model_name="BaselineLR"
    )
    plot_y_vs_yhat(
        y_labeled,
        base_oof,
        os.path.join(PLOT_DIR, "y_vs_yhat_baselineLR.png"),
        title="Baseline LR (Task4 script) y vs y_hat"
    )

    modelA = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    folds_A, oof_A, mean_A, std_A = evaluate_model_with_kfold(
        modelA, X_all, y_all, c_all, labeled_mask,
        cv_splits=5,
        model_name="SemiSup-Impute-LR"
    )
    plot_y_vs_yhat(
        y_labeled,
        oof_A,
        os.path.join(PLOT_DIR, "y_vs_yhat_semi_impute_LR.png"),
        title="Task4 Model A (Semi-supervised Impute + LR)"
    )

    iso_scaler = StandardScaler()
    X_all_scaled = iso_scaler.fit_transform(X_all)

    n_components = 2
    iso = Isomap(n_components=n_components)
    iso.fit(X_all_scaled)
    print(f"Fitted Isomap with n_components={n_components} on ALL data")


    modelB = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        FrozenTransformer(iso),
        LinearRegression()
    )

    folds_B, oof_B, mean_B, std_B = evaluate_model_with_kfold(
        modelB, X_all, y_all, c_all, labeled_mask,
        cv_splits=5,
        model_name="Isomap-SemiSup-LR"
    )
    plot_y_vs_yhat(
        y_labeled,
        oof_B,
        os.path.join(PLOT_DIR, "y_vs_yhat_isomap_semi_LR.png"),
        title="Task4 Model B (Isomap semi-sup + LR)"
    )

    summary_rows = [
        {
            "model": "BaselineLR_Task4Script",
            "description": "Supervised LR on imputed features",
            "mean_error": base_mean,
            "std_error": base_std,
            "min_error": float(np.min(base_folds)),
            "max_error": float(np.max(base_folds))
        },
        {
            "model": "SemiSup-Impute-LR",
            "description": "Imputation using labeled+unlabeled, LR on labeled",
            "mean_error": mean_A,
            "std_error": std_A,
            "min_error": float(np.min(folds_A)),
            "max_error": float(np.max(folds_A))
        },
        {
            "model": "Isomap-SemiSup-LR",
            "description": "Isomap on labeled+unlabeled + LR",
            "mean_error": mean_B,
            "std_error": std_B,
            "min_error": float(np.min(folds_B)),
            "max_error": float(np.max(folds_B))
        }
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=["mean_error", "std_error"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print("Saved Task 4 CV summary:", SUMMARY_CSV)
    print(summary_df)

    model_scores = [
        ("BaselineLR_Task4Script", baseline_pipe, base_mean, base_std),
        ("SemiSup-Impute-LR", modelA, mean_A, std_A),
        ("Isomap-SemiSup-LR", modelB, mean_B, std_B),
    ]
    best_name, best_model, _, _ = min(model_scores, key=lambda t: (t[2], t[3]))
    print(f"Best model according to CV: {best_name}")

    best_model.fit(X_labeled, y_labeled)
    train_preds = best_model.predict(X_labeled)
    train_c = c_labeled
    train_cMSE = error_metric(y_labeled, train_preds, train_c)
    print(f"{best_name} final train cMSE on labeled data: {train_cMSE:.6f}")

    X_test_features = X_test_pre.drop(columns=[ID_COL], errors="ignore")

    test_preds = best_model.predict(X_test_features)

    submission_path = get_next_submission_filename(
        task_num=4,
        base_name="semisupervised-submission"
    )

    submission = sample_sub.copy()
    if submission.shape[1] >= 2:
        target_col = submission.columns[1]
        submission[target_col] = test_preds
    else:
        ids = submission[ID_COL].values if ID_COL in submission.columns else np.arange(len(test_preds))
        submission = pd.DataFrame({ID_COL: ids, TARGET_COL: test_preds})

    submission.to_csv(submission_path, index=False)
    print("Saved Task 4 Kaggle submission to:", submission_path)

    print("\n--- Task 4 Summary ---")
    print("Labeled rows:", labeled_mask.sum())
    print("Unlabeled rows:", unlabeled_mask.sum())
    print("Final best model:", best_name)
    print("Train cMSE (labeled):", train_cMSE)
    print("CV summary:\n", summary_df.head())

if __name__ == "__main__":
    main()