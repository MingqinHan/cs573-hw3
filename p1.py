import numpy as np

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics


def load_data(path="datahw3.npz"):
    """
    Load the npz file and return two datasets:
    D1 uses (train_dat_y,  val_dat_y)
    D2 uses (train_dat_y2, val_dat_y2)
    """
    data = np.load(path)

    X_train = data["train_dat_x"]
    y_train = data["train_dat_y"]
    X_val = data["val_dat_x"]
    y_val = data["val_dat_y"]

    y_train2 = data["train_dat_y2"]
    y_val2 = data["val_dat_y2"]

    return (X_train, y_train, X_val, y_val,
            X_train, y_train2, X_val, y_val2)


def run_bagging(X_train, y_train, X_test, y_test, dataset_name):
    print(f"\n=== Bagging on {dataset_name} ===")
    n_list = [2, 10, 50, 75, 100]

    for n in n_list:
        # Base estimator: SVM classifier
        base_svm = SVC(probability=True, random_state=0)

        clf = BaggingClassifier(
            base_svm,          # estimator
            n_estimators=n,
            random_state=0
        )

        clf.fit(X_train, y_train)

        # zero-one loss = 1 - accuracy
        y_pred = clf.predict(X_test)
        loss = metrics.zero_one_loss(y_test, y_pred)

        # AUC (binary classification, y in {0,1})
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = metrics.roc_auc_score(y_test, y_proba)

        print(f"n_estimators={n:3d}: loss={loss:.3f}, AUC={auc:.3f}")


def run_boosting(X_train, y_train, X_test, y_test, dataset_name):
    print(f"\n=== Boosting on {dataset_name} ===")
    n_list = [2, 10, 50, 75, 100]

    for n in n_list:
        # Base estimator: decision stump
        base_tree = DecisionTreeClassifier(max_depth=1, random_state=0)

        clf = AdaBoostClassifier(
            base_tree,         # estimator
            n_estimators=n,
            algorithm="SAMME",
            random_state=0
        )

        clf.fit(X_train, y_train)

        # zero-one loss
        y_pred = clf.predict(X_test)
        loss = metrics.zero_one_loss(y_test, y_pred)

        # AUC
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = metrics.roc_auc_score(y_test, y_proba)

        print(f"n_estimators={n:3d}: loss={loss:.3f}, AUC={auc:.3f}")


def main():
    data_path = "datahw3.npz"

    (X1_tr, y1_tr, X1_te, y1_te,
     X2_tr, y2_tr, X2_te, y2_te) = load_data(data_path)

    # Dataset D1: labels y
    run_bagging(X1_tr, y1_tr, X1_te, y1_te, "D1 (y)")
    run_boosting(X1_tr, y1_tr, X1_te, y1_te, "D1 (y)")

    # Dataset D2: labels y2
    run_bagging(X2_tr, y2_tr, X2_te, y2_te, "D2 (y2)")
    run_boosting(X2_tr, y2_tr, X2_te, y2_te, "D2 (y2)")


if __name__ == "__main__":
    main()

