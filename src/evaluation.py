from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_model(
    model,
    X_test,
    y_test,
    training_time: float = None,
) -> dict:
    """Evaluate a fitted classifier and return a results dictionary.

    Parameters
    ----------
    model : fitted sklearn estimator
        A trained classifier with a predict method.
    X_test : sparse matrix
        TF-IDF feature matrix for the test set.
    y_test : array-like
        True binary labels (1 = positive, 0 = negative).
    training_time : float, optional
        Elapsed training time in seconds, captured at fit time.

    Returns
    -------
    dict
        Keys: accuracy, precision_macro, recall_macro, f1_macro,
        precision_per_class, recall_per_class, f1_per_class,
        confusion_matrix, y_pred, training_time (seconds or None).
    """
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1]
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    return {
        'accuracy':            accuracy_score(y_test, y_pred),
        'precision_macro':     p_macro,
        'recall_macro':        r_macro,
        'f1_macro':            f1_macro,
        'precision_per_class': precision,
        'recall_per_class':    recall,
        'f1_per_class':        f1,
        'confusion_matrix':    confusion_matrix(y_test, y_pred),
        'y_pred':              y_pred,
        'training_time':       training_time,
    }
