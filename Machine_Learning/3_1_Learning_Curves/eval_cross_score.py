def scores_table_classification(scores: dict, model_name: str = "Model", digits: int = 4) -> str:
    """
    Formate un tableau avec Accuracy, Precision, Recall, F1, ROC AUC
    à partir d'un dict 'scores' retourné par cross_validate (avec return_train_score=True)
    ou d'un dict cv_results_ de GridSearchCV.
    """
    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC AUC", "roc_auc"),
    ]

    def _mean_or_none(key: str):
        if key in scores:
            val = scores[key]
            try:
                return float(np.mean(val))
            except Exception:
                return float(val)
        return None

    def _get(split: str, mkey: str):
        # cross_validate -> 'train_f1' / 'test_f1'
        # GridSearchCV    -> 'mean_train_f1' / 'mean_test_f1'
        for k in (f"{split}_{mkey}", f"mean_{split}_{mkey}"):
            v = _mean_or_none(k)
            if v is not None:
                return v
        return None

    def _fmt(x):
        return f"{x:.{digits}f}" if x is not None else "-"

    title = f"--- Résultats pour le modèle : {model_name} ---"
    header = "Métrique   | Train    | Test"
    bar = "-" * max(len(title), len(header))

    lines = [title, header, bar]
    for label, key in metrics:
        tr = _get("train", key)
        te = _get("test",  key)
        lines.append(f"{label:<10} | {_fmt(tr):>8} | {_fmt(te):>8}")
    lines.append(bar)
    return "\n".join(lines)



def scores_table_regression(scores: dict, model_name: str = "Model", digits: int = 4) -> str:
    """
    Formate un tableau avec Variance, -MSE, -RMSE, et r2
    à partir d'un dict 'scores' retourné par cross_validate (avec return_train_score=True)
    ou d'un dict cv_results_ de GridSearchCV.
    """
    metrics = [
        ("explained_variance", "explained_variance"),
        ("neg_mean_squared_error", "neg_mean_squared_error"),
        ("neg_root_mean_squared_error", "neg_root_mean_squared_error"),
        ("r2", "r2"),
    ]

    def _mean_or_none(key: str):
        if key in scores:
            val = scores[key]
            try:
                return float(np.mean(val))
            except Exception:
                return float(val)
        return None

    def _get(split: str, mkey: str):
        # cross_validate -> 'train_f1' / 'test_f1'
        # GridSearchCV    -> 'mean_train_f1' / 'mean_test_f1'
        for k in (f"{split}_{mkey}", f"mean_{split}_{mkey}"):
            v = _mean_or_none(k)
            if v is not None:
                return v
        return None

    def _fmt(x):
        return f"{x:.{digits}f}" if x is not None else "-"

    title = f"--- Résultats pour le modèle : {model_name} ---"
    header = "Métrique   | Train    | Test"
    bar = "-" * max(len(title), len(header))

    lines = [title, header, bar]
    for label, key in metrics:
        tr = _get("train", key)
        te = _get("test",  key)
        lines.append(f"{label:<10} | {_fmt(tr):>8} | {_fmt(te):>8}")
    lines.append(bar)
    return "\n".join(lines)