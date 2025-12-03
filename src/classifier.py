import numpy as np


from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, LeaveOneOut

def make_features_row(a, c, bins=10, rng=(0, 1), density=True):
    def _hist(seq):
        # treat None/NaN as empty
        if seq is None or (isinstance(seq, float) and np.isnan(seq)):
            arr = np.array([], dtype=float)
        else:
            arr = np.asarray(seq, dtype=float).ravel()

        if arr.size == 0:
            h = np.zeros(bins, dtype=float)
        else:
            h, _ = np.histogram(arr, bins=bins, range=rng, density=True)
            h = h.astype(float)

        return h

    h_action  = _hist(a)
    h_content = _hist(c)
    return np.concatenate([h_action, h_content])

def classifier(df, task):
    if(task == "automation_detection"):
        min_class_size = df['user_class'].value_counts().min()
        balanced_df = (
            df.groupby('user_class', group_keys=False)
            .apply(lambda x: x.sample(min_class_size, random_state=42))
            .sample(frac=1, random_state=42)  # shuffle once
            .reset_index(drop=True)
        )

        bins = 20
        X = np.vstack([
            make_features_row(a, c, bins=bins)
            for a, c in zip(balanced_df["action_changes_list"], balanced_df["content_changes_list"])
        ])
        y = balanced_df['user_class'].values

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        min_train_fold = X.shape[0] - X.shape[0] // cv.n_splits
        max_k = min(10, max(1, min_train_fold - 1))

        best_score, best_k, best_cv_pred = -1.0, None, None
        for k in range(1, max_k + 1):
            clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', algorithm='brute')
            y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1, method='predict')
            macro_f1 = f1_score(y, y_pred, average='macro')
            if macro_f1 > best_score or (macro_f1 == best_score and (best_k is None or k < best_k)):
                best_score, best_k, best_cv_pred = macro_f1, k, y_pred

        print("automation  detection results")
        print(classification_report(y, best_cv_pred))  
    else:
        min_class_size = df['user_class'].value_counts().min()
        newdf = df.groupby('user_class').sample(n=min_class_size, random_state=42)

        bins = 10
        X = np.vstack([
            make_features_row(a, c, bins=bins)
            for a, c in zip(newdf["action_changes_list"], newdf["content_changes_list"])
        ])
        y = newdf['user_class'].values 
        
        loo = LeaveOneOut()
        best_k = None
        best_score = -1.0
        best_y = None

        max_k = min(10, max(1, X.shape[0] - 1))
        for k in range(1, max_k):
            clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', algorithm='brute')
            y_pred = cross_val_predict(clf, X, y, cv=loo, n_jobs=-1, method='predict')
            macro_f1 = f1_score(y, y_pred, average='macro')  # global macro-F1 across all LOOCV folds
            if macro_f1 > best_score or (macro_f1 == best_score and (best_k is None or k < best_k)):
                best_score = macro_f1
                best_k = k
                best_y = y_pred
                
        print("coordination detection results")
        print(classification_report(y, best_y, output_dict=True))

    return best_score, min_class_size