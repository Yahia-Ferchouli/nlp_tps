
from sklearn.feature_extraction.text import CountVectorizer
import joblib

TEXT_COLUMN = 'video_name'
TARGET_COLUMN = 'is_comic'


def make_features(df, save_vectorizer=True, vectorizer_path='vectorizer.pkl'):
    """Create features from the DataFrame and return them."""

    try:
        vectorizer = joblib.load(vectorizer_path)  # Charger le vectorizer s'il existe
        X = vectorizer.transform(df[TEXT_COLUMN])  # Transformer les données
    except (FileNotFoundError, ValueError):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df[TEXT_COLUMN])  # Pour l'entraînement

        if save_vectorizer:
            joblib.dump(vectorizer, vectorizer_path)  # Sauvegarder le vectorizer

    y = df[TARGET_COLUMN]  # La cible
    return X, y
