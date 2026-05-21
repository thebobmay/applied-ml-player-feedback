import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text: str) -> str:
    """Clean and normalize review text for vectorization.

    Steps applied in order:
    1. Lowercase
    2. Strip HTML tags
    3. Strip BBCode markup (e.g. [b], [url=...])
    4. Strip URLs
    5. Remove non-ASCII characters
    6. Remove punctuation
    7. Collapse extra whitespace

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    str
        Cleaned, normalized text ready for TF-IDF vectorization.
    """
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[/?[a-z]+(=[^\]]*)?\]', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_feature_matrix(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer = None,
) -> tuple:
    """Vectorize review text using TF-IDF and return feature matrix, labels, and vectorizer.

    When called without a fitted vectorizer, fits a new TfidfVectorizer on the
    provided DataFrame and returns the fitted vectorizer as the third element.
    When called with an existing vectorizer, transforms without refitting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'review_clean' (preprocessed text) and 'voted_up' columns.
    vectorizer : TfidfVectorizer, optional
        A pre-fitted TfidfVectorizer. Pass None to fit a new one (training set).
        Pass the fitted vectorizer to transform without refitting (test set).

    Returns
    -------
    tuple
        (X, y, vectorizer) where X is a sparse TF-IDF matrix, y is a binary label
        array (1 = positive, 0 = negative), and vectorizer is the fitted TfidfVectorizer.
    """
    fit = vectorizer is None
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=50_000,
            min_df=5,
            max_df=0.95,
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(df['review_clean'])
    else:
        X = vectorizer.transform(df['review_clean'])
    y = df['voted_up'].astype(int).values
    action = 'Fit + transformed' if fit else 'Transformed'
    print(f"{action}: {X.shape[0]:,} reviews x {X.shape[1]:,} features")
    return X, y, vectorizer
