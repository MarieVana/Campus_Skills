from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import numpy as np
import pandas as pd

def _align_X_for_model(model, X):
    """Assure que X a les mêmes colonnes/shape que lors du fit (utile avec DataFrame)."""
    import pandas as pd, numpy as np
    if hasattr(model, "feature_names_in_") and isinstance(X, pd.DataFrame):
        X = X[list(model.feature_names_in_)]
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def eval_metrics(model, X_train, y_train, X_test, y_test, average="weighted", class_names=None):
    """
    Retourne (metrics_df, cm_test_df, cm_train_df)
    - metrics_df : lignes ['Train','Test'] avec Accuracy, Precision, Recall, F1, Support
    - cm_test_df : matrice de confusion du Test (étiquetée)
    - cm_train_df : matrice de confusion du Train (étiquetée)
    """
    # S'assurer que X correspond à ce que le modèle attend
    X_train = _align_X_for_model(model, X_train)
    X_test  = _align_X_for_model(model, X_test)

    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Labels observés (ordre stable)
    labels = np.unique(np.concatenate([np.unique(y_train), np.unique(y_test)]))

    # Noms de classes
    if class_names is None or len(class_names) != len(labels):
        class_names = [str(l) for l in labels]
    name_by_label = {lab: name for lab, name in zip(labels, class_names)}

    # --- Global TRAIN
    acc_tr = accuracy_score(y_train, y_pred_train)
    p_tr, r_tr, f1_tr, _ = precision_recall_fscore_support(
        y_train, y_pred_train, average=average, zero_division=0
    )

    # --- Global TEST
    acc_te = accuracy_score(y_test, y_pred_test)
    p_te, r_te, f1_te, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average=average, zero_division=0
    )

    metrics_df = pd.DataFrame([
        {"Split":"Train","Accuracy":acc_tr,"Precision":p_tr,"Recall":r_tr,"F1":f1_tr,"Support":len(y_train)},
        {"Split":"Test", "Accuracy":acc_te,"Precision":p_te,"Recall":r_te,"F1":f1_te,"Support":len(y_test)},
    ])

    # --- Matrices de confusion
    cm_test  = confusion_matrix(y_test,  y_pred_test,  labels=labels)
    cm_train = confusion_matrix(y_train, y_pred_train, labels=labels)

    cm_test_df = pd.DataFrame(
        cm_test,
        index=[f"Vrai {name_by_label[l]}" for l in labels],
        columns=[f"Prédit {name_by_label[l]}" for l in labels]
    )
    cm_train_df = pd.DataFrame(
        cm_train,
        index=[f"Vrai {name_by_label[l]}" for l in labels],
        columns=[f"Prédit {name_by_label[l]}" for l in labels]
    )

    return metrics_df, cm_test_df, cm_train_df