from sklearn.ensemble import RandomForestClassifier

def make_model():
    # Crée et retourne un modèle Random Forest
    model = RandomForestClassifier(random_state=42)  # On fixe le random_state pour la reproductibilité
    return model