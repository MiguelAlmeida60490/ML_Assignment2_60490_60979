import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.utils import resample

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

ENABLE_SELF_TRAINING = True
SELF_TRAIN_MAX_ITERS = 3
SELF_TRAIN_ADD_FRACTION = 0.10
BOOTSTRAP_ENSEMBLE_SIZE = 7
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

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

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.reindex(columns=X_test.columns, fill_value=0)

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

def make_polynomial_pipeline(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

def make_knn_pipeline(k):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])

def bootstrap_ensemble_predictions(base_pipeline, X_labeled, y_labeled, X_unlabeled, n_estimators=BOOTSTRAP_ENSEMBLE_SIZE, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    preds_collection = []
    for i in range(n_estimators):
        Xb, yb = resample(X_labeled, y_labeled, replace=True, random_state=rng.randint(0, 1_000_000))
        p = clone(base_pipeline)
        p.fit(Xb, yb)
        preds_collection.append(p.predict(X_unlabeled))
    preds = np.vstack(preds_collection)
    return preds.mean(axis=0), preds.std(axis=0)

def self_training_pipeline(base_pipeline, X_all, y_all, c_all, labeled_mask, unlabeled_mask,max_iters=SELF_TRAIN_MAX_ITERS, add_fraction=SELF_TRAIN_ADD_FRACTION):
    labeled_idx = list(np.where(labeled_mask)[0])
    unlabeled_idx = list(np.where(unlabeled_mask)[0])

    X = X_all
    y = y_all.copy()
    c = c_all.copy()

    if len(unlabeled_idx) == 0:
        base_pipeline.fit(X.iloc[labeled_idx], y[labeled_idx])
        return base_pipeline, X, y, c

    pipeline = clone(base_pipeline)
    for it in range(max_iters):
        pipeline.fit(X.iloc[labeled_idx], y[labeled_idx])
        if len(unlabeled_idx) == 0:
            break
        X_unl = X.iloc[unlabeled_idx]
        preds_mean, preds_std = bootstrap_ensemble_predictions(pipeline, X.iloc[labeled_idx], y[labeled_idx], X_unl)

        n_add = max(1, int(len(unlabeled_idx) * add_fraction))
        pick_local = np.argsort(preds_std)[:n_add]
        pick_global = [unlabeled_idx[i] for i in pick_local]

        for pl, pg in zip(pick_local, pick_global):
            y[pg] = float(preds_mean[pl])
            c[pg] = 0 
            labeled_idx.append(pg)

        unlabeled_idx = [idx for idx in unlabeled_idx if idx not in pick_global]
        print(f" Self-train iter {it+1}: added {len(pick_global)} pseudo-labels; unlabeled left {len(unlabeled_idx)}")

    pipeline.fit(X.iloc[labeled_idx], y[labeled_idx])
    print("Self-training finished. Final labeled count:", len(labeled_idx))
    return pipeline, X, y, c

def run_model_selection(X, y, c, labeled_mask, cv_splits=5, poly_degrees=(1,2,3), knn_neighbors=(3,5,7,9)):
    labeled_indices = np.where(labeled_mask)[0]
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for deg in poly_degrees:
        pipe = make_polynomial_pipeline(deg)
        fold_errors, _ = cv_evaluate_pipeline(pipe, X, y, c, kf, labeled_indices)
        results.append(dict(model="polynomial", param=deg, fold_errors=fold_errors,
                            mean_error=np.mean(fold_errors), std_error=np.std(fold_errors),
                            min_error=np.min(fold_errors), max_error=np.max(fold_errors)))
    for k in knn_neighbors:
        pipe = make_knn_pipeline(k)
        fold_errors, _ = cv_evaluate_pipeline(pipe, X, y, c, kf, labeled_indices)
        results.append(dict(model="knn", param=k, fold_errors=fold_errors,
                            mean_error=np.mean(fold_errors), std_error=np.std(fold_errors),
                            min_error=np.min(fold_errors), max_error=np.max(fold_errors)))
    rows = [{"model": r["model"], "param": r["param"], "mean_error": r["mean_error"], "std_error": r["std_error"],
             "min_error": r["min_error"], "max_error": r["max_error"]} for r in results]
    summary_df = pd.DataFrame(rows).sort_values(by=["mean_error", "std_error"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print("Saved CV summary:", SUMMARY_CSV)
    return results, summary_df

def select_best_model_from_summary(results):
    return min(results, key=lambda r: (r["mean_error"], r["std_error"]))

def train_final_and_generate_submission(best_entry, X, y, c, X_test_df, sample_sub, out_filename=OUT_SUBMISSION):
    if best_entry["model"] == "polynomial":
        final_pipe = make_polynomial_pipeline(best_entry["param"])
    else:
        final_pipe = make_knn_pipeline(best_entry["param"])

    train_mask = ~pd.isna(y)
    final_pipe.fit(X.loc[train_mask, :], y[train_mask])

    train_preds = final_pipe.predict(X.loc[train_mask, :])
    train_err = error_metric(y[train_mask], train_preds, c[train_mask])
    print("Final train cMSE:", train_err)

    Xt = X_test_df.copy()
    if ID_COL in Xt.columns:
        Xt = Xt.drop(columns=[ID_COL])
    test_preds = final_pipe.predict(Xt)

    sub = sample_sub.copy()
    cols = sub.columns.tolist()
    if len(cols) >= 2:
        sub[cols[1]] = test_preds
    else:
        ids = sub[ID_COL].values if ID_COL in sub.columns else np.arange(len(test_preds))
        sub = pd.DataFrame({ID_COL: ids, TARGET_COL: test_preds})
    sub.to_csv(out_filename, index=False)
    print("Saved submission:", out_filename)
    return final_pipe, train_err

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

    poly_degrees = (1,2,3)
    knn_neighbors = (3,5,7,9)
    results, summary_df = run_model_selection(X_all, y_all, c_all, labeled_mask, cv_splits=5, poly_degrees=poly_degrees, knn_neighbors=knn_neighbors)
    best = select_best_model_from_summary(results)
    print("Best model:", best)

    if ENABLE_SELF_TRAINING:
        print("Starting self-training...")
        base_pipe = make_polynomial_pipeline(best["param"]) if best["model"]=="polynomial" else make_knn_pipeline(best["param"])
        trained_pipe, X_aug, y_aug, c_aug = self_training_pipeline(base_pipe, X_all, y_all, c_all, labeled_mask, unlabeled_mask, max_iters=SELF_TRAIN_MAX_ITERS, add_fraction=SELF_TRAIN_ADD_FRACTION)
    else:
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
