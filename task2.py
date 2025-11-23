import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

TRAIN_PATH = "train_data.csv"
TEST_PATH = "test_data.csv"
SAMPLE_PATH = "sample_submission.csv"

ID_COL = "PatientID"
TARGET_COL = "SurvivalTime"
CENSOR_COL = "Censored"

OUT_SUBMISSION = "Nonlinear-submission-01.csv"
PLOT_DIR = "plots_task2"
SUMMARY_CSV = "task2_cv_summary.csv"
os.makedirs(PLOT_DIR, exist_ok=True)

# Semi-supervised disabled (we removed self-training in the clean version)
ENABLE_SELF_TRAINING = False

# Hyperparameters / settings
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def error_metric(y, y_hat, c):
    """
    censored Mean Squared Error (cMSE):
      - for uncensored (c==0): (y - y_hat)^2
      - for censored   (c==1): max(0, y - y_hat)^2  (no penalty if y_hat >= y)
    """
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
    return train, test, sample

def standardize_id_names(train_df, test_df, sample_df):
    df_tr = train_df
    df_te = test_df
    df_s = sample_df
    found = find_id_col(df_tr) or find_id_col(df_te) or find_id_col(df_s)
    if found is None:
        df_tr = df_tr.copy(); df_te = df_te.copy()
        df_tr[ID_COL] = np.arange(len(df_tr))
        df_te[ID_COL] = np.arange(len(df_te))
        if df_s.shape[1] >= 1:
            df_s = df_s.copy(); df_s = df_s.rename(columns={df_s.columns[0]: ID_COL})
        return df_tr, df_te, df_s
    if found != ID_COL:
        if found in df_tr.columns:
            df_tr = df_tr.rename(columns={found: ID_COL})
        if found in df_te.columns:
            df_te = df_te.rename(columns={found: ID_COL})
        if found in df_s.columns:
            df_s = df_s.rename(columns={found: ID_COL})
    return df_tr, df_te, df_s

def produce_missingness_plots(train_df, out_dir=PLOT_DIR):
    plt.figure(figsize=(10,4)); msno.bar(train_df); plt.title("Missing values - bar"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "missing_bar.png")); plt.close()
    plt.figure(figsize=(12,6)); msno.matrix(train_df); plt.savefig(os.path.join(out_dir, "missing_matrix.png")); plt.close()
    plt.figure(figsize=(10,8)); msno.heatmap(train_df); plt.savefig(os.path.join(out_dir, "missing_heatmap.png")); plt.close()
    try:
        plt.figure(figsize=(10,8)); msno.dendrogram(train_df); plt.savefig(os.path.join(out_dir, "missing_dendrogram.png")); plt.close()
    except Exception as e:
        print("dendrogram skipped:", e)

def preprocess_features(train_df, test_df):
    t = train_df.copy()
    s = test_df.copy()

    if ID_COL not in t.columns:
        t[ID_COL] = np.arange(len(t))
    test_has_id = ID_COL in s.columns
    if not test_has_id:
        s = s.copy()
        s[ID_COL] = np.arange(len(s))

    excluded = {TARGET_COL, CENSOR_COL}
    feature_cols = [c for c in t.columns if c not in excluded and c != ID_COL]

    num_cols = t[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    if num_cols:
        num_imp = SimpleImputer(strategy="mean")
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

    # Align test to train (one-directional)
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

    print("prepare_X_y_c: n_rows =", len(X), "labeled =", labeled_mask.sum(), "unlabeled =", unlabeled_mask.sum())
    return X.reset_index(drop=True), y, c, labeled_mask, unlabeled_mask

def cv_evaluate_pipeline(pipeline, X, y, c, kfold, labeled_indices):
    labeled_indices = np.asarray(labeled_indices)
    n = len(labeled_indices)
    fold_errors = []
    oof_preds = np.zeros(n, dtype=float)

    for fold_idx, (tr_local, te_local) in enumerate(kfold.split(labeled_indices)):
        tr_idx = labeled_indices[tr_local]
        te_idx = labeled_indices[te_local]

        X_tr = X.iloc[tr_idx]; X_te = X.iloc[te_idx]
        y_tr = y[tr_idx]; y_te = y[te_idx]
        c_tr = c[tr_idx]; c_te = c[te_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_te)
        err = error_metric(y_te, preds, c_te)
        fold_errors.append(err)
        oof_preds[te_local] = preds
        print(f" Fold {fold_idx+1}/{kfold.get_n_splits()} cMSE={err:.6f}")

    return fold_errors, oof_preds

def make_polynomial_pipeline(degree, alpha=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

def make_knn_pipeline(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])


def run_model_selection(X, y, c, labeled_mask, cv_splits=5,
                        poly_degrees=(1,2,3),
                        ridge_alphas=(0.1, 1.0, 10.0, 50.0),
                        knn_neighbors=(3,5,7,9,15)):
    """
    Cross-validate polynomial(Ridge) variants and KNN variants.
    Returns results list (dictionaries) and a summary_df sorted by mean_error.
    """
    labeled_indices = np.where(labeled_mask)[0]
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    results = []

    # Ridge polynomial models
    for deg in poly_degrees:
        for alpha in ridge_alphas:
            pipe = make_polynomial_pipeline(deg, alpha)
            fold_errors = []

            for tr_loc, te_loc in kf.split(labeled_indices):
                tr = labeled_indices[tr_loc]
                te = labeled_indices[te_loc]

                pipe.fit(X.iloc[tr], y[tr])
                preds = pipe.predict(X.iloc[te])
                err = error_metric(y[te], preds, c[te])
                fold_errors.append(err)

            results.append({
                "model": "polynomial",
                "degree": deg,
                "alpha": alpha,
                "mean_error": np.mean(fold_errors),
                "std_error": np.std(fold_errors),
                "fold_errors": fold_errors
            })
            print(f"[CV] poly deg={deg}, alpha={alpha}, mean={np.mean(fold_errors):.4f}")

    # KNN models
    for k in knn_neighbors:
        pipe = make_knn_pipeline(k)
        fold_errors = []

        for tr_loc, te_loc in kf.split(labeled_indices):
            tr = labeled_indices[tr_loc]
            te = labeled_indices[te_loc]

            pipe.fit(X.iloc[tr], y[tr])
            preds = pipe.predict(X.iloc[te])
            err = error_metric(y[te], preds, c[te])
            fold_errors.append(err)

        results.append({
            "model": "knn",
            "k": k,
            "mean_error": np.mean(fold_errors),
            "std_error": np.std(fold_errors),
            "fold_errors": fold_errors
        })
        print(f"[CV] knn k={k}, mean={np.mean(fold_errors):.4f}")

    # Build summary
    summary_rows = []
    for r in results:
        if r["model"] == "polynomial":
            param_str = f"deg={r['degree']}, alpha={r['alpha']}"
        else:
            param_str = f"k={r['k']}"

        summary_rows.append({
            "model": r["model"],
            "param": param_str,
            "mean_error": r["mean_error"],
            "std_error": r["std_error"],
            "min_error": np.min(r["fold_errors"]),
            "max_error": np.max(r["fold_errors"])
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["mean_error", "std_error"]
    ).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print("Saved CV summary:", SUMMARY_CSV)

    return results, summary_df

def select_best_model_from_summary(results):
    """Return the dictionary entry from results with smallest mean_error (then std)."""
    return min(results, key=lambda r: (r["mean_error"], r["std_error"]))

def train_final_and_generate_submission(best_entry, X, y, c, X_test_df, sample_sub, out_filename):
    """
    Train the selected best_entry on ALL labeled data and generate a Kaggle-style submission.
    best_entry is an element from results returned by run_model_selection.
    """
    # Build best model
    if best_entry["model"] == "polynomial":
        pipe = make_polynomial_pipeline(best_entry["degree"], best_entry["alpha"])
    else:
        pipe = make_knn_pipeline(best_entry["k"])

    # Train only on labeled data
    labeled = ~pd.isna(y)
    pipe.fit(X.loc[labeled], y[labeled])

    # Report training cMSE
    preds_train = pipe.predict(X.loc[labeled])
    train_err = error_metric(y[labeled], preds_train, c[labeled])
    print("Final train cMSE:", train_err)

    # Predict test
    Xt = X_test_df.drop(columns=[ID_COL], errors="ignore")
    test_preds = pipe.predict(Xt)

    # Write submission
    sub = sample_sub.copy()
    if sub.shape[1] >= 2:
        target_col = sub.columns[1]
        sub[target_col] = test_preds
    else:
        # fallback: create proper submission with ID and target
        ids = sub[ID_COL].values if ID_COL in sub.columns else np.arange(len(test_preds))
        sub = pd.DataFrame({ID_COL: ids, TARGET_COL: test_preds})

    sub.to_csv(out_filename, index=False)
    print("Saved submission:", out_filename)
    return pipe, train_err

def plot_cv_summary(summary_df):
    plt.figure(figsize=(8,5))
    s = summary_df.copy()
    s["param"] = s["param"].astype(str)
    sns.barplot(data=s, x="model", y="mean_error", hue="param")
    plt.title("CV mean cMSE by model & param")
    plt.tight_layout()
    fn = os.path.join(PLOT_DIR, "cv_mean_errors.png")
    plt.savefig(fn); plt.close(); print("Saved:", fn)

def plot_y_vs_yhat(y_true, y_pred, out_path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.savefig(out_path); plt.close(); print("Saved:", out_path)
    
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

    print("Loaded shapes:", train_df.shape, test_df.shape, sample_sub.shape)
    produce_missingness_plots(train_df)

    X_train_pre, X_test_pre, feature_cols = preprocess_features(train_df, test_df)

    try:
        numeric_cols = X_train_pre.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != ID_COL][:6]
        if numeric_cols:
            if TARGET_COL in train_df.columns:
                tmp = pd.concat([X_train_pre[numeric_cols].reset_index(drop=True), train_df[[TARGET_COL]].reset_index(drop=True)], axis=1)
                sns.pairplot(tmp.dropna(subset=[TARGET_COL]), vars=numeric_cols + [TARGET_COL])
            else:
                sns.pairplot(X_train_pre[numeric_cols])
            plt.savefig(os.path.join(PLOT_DIR, "pairplot_numeric_limited.png")); plt.close()
    except Exception as e:
        print("Pairplot skipped:", e)

    X_all, y_all, c_all, labeled_mask, unlabeled_mask = prepare_X_y_c(train_df, X_train_pre)

    # Model selection and hyperparameter ranges
    poly_degrees = (1,2,3)
    knn_neighbors = (3,5,7,9)

    results, summary_df = run_model_selection(
        X_all, y_all, c_all, labeled_mask,
        cv_splits=5,
        poly_degrees=poly_degrees,
        ridge_alphas=(0.1, 1.0, 10.0, 50.0),
        knn_neighbors=knn_neighbors
    )

    best = select_best_model_from_summary(results)
    print("Summary_df:", summary_df)
    print("Best model (selected):", best)

    # We disabled self-training in the clean version; use X_all/y_all as is
    X_aug, y_aug, c_aug = X_all, y_all, c_all

    submission_path = get_next_submission_filename(
        task_num=2,
        base_name="Nonlinear-submission"
    )

    final_pipeline, train_err = train_final_and_generate_submission(
        best,
        X_aug,
        y_aug,
        c_aug,
        X_test_pre,
        sample_sub,
        out_filename=submission_path
    )

    plot_cv_summary(summary_df)
    try:
        mask = ~pd.isna(y_aug)
        plot_y_vs_yhat(y_aug[mask], final_pipeline.predict(X_aug.loc[mask,:]), os.path.join(PLOT_DIR, "y_vs_yhat_final.png"))
    except Exception as e:
        print("y vs yhat plot skipped:", e)

    print("\n--- Summary ---")
    missing_cols = [c for c in train_df.columns if train_df[c].isna().any() and c not in {TARGET_COL, CENSOR_COL}]
    print("Missing columns (original):", missing_cols)
    print("Total train rows:", train_df.shape[0])
    print("Labeled for CV:", labeled_mask.sum(), "Unlabeled:", unlabeled_mask.sum())
    print("Final training rows (after pseudo-labeling):", (~pd.isna(y_aug)).sum())
    print("Final feature count:", X_aug.shape[1])
    print("Train cMSE:", train_err)
    print("Saved CV summary (top rows):"); print(summary_df.head())

if __name__ == "__main__":
    main()