import click
import joblib
from src.data import make_dataset
from src.features import make_features
from src.models import make_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@click.group()
def cli():
    pass

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Fichier de données d'entraînement")
@click.option("--model_dump_filename", default="models/model.json", help="Fichier pour sauvegarder le modèle")
def train(input_filename, model_dump_filename):
    """Entraîne le modèle et l'enregistre dans un fichier."""
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = make_model()
    model.fit(X, y)

    joblib.dump(model, model_dump_filename)
    click.echo(f'Modèle sauvegardé dans {model_dump_filename}')

@click.command()
@click.option("--input_filename", default="data/processed/test.csv", help="Fichier avec de nouveaux titres de vidéos pour la prédiction")
@click.option("--model_dump_filename", default="models/model.json", help="Fichier avec le modèle sauvegardé")
@click.option("--output_filename", default="data/processed/predictions.csv", help="Fichier de sortie pour les prédictions")
def predict(input_filename, model_dump_filename, output_filename):
    """Fait des prédictions sur de nouvelles données."""
    try:
        # Charger le modèle entraîné
        model = joblib.load(model_dump_filename)

        # Charger les données d'entrée
        df = make_dataset(input_filename)

        # Créer les caractéristiques à partir des données d'entrée
        X, _ = make_features(df, save_vectorizer=False)  # Ne pas sauvegarder le vectorizer ici

        # Faire des prédictions
        predictions = model.predict(X)

        # Sauvegarder les prédictions dans un DataFrame et l'exporter en CSV
        df['is_comic'] = predictions
        df.to_csv(output_filename, index=False)

        # Afficher quelques prédictions dans la console
        click.echo(f'Prédictions sauvegardées dans {output_filename}')
        click.echo('Exemples de prédictions :')
        click.echo(df[['video_name', 'is_comic']].head())  # Afficher les premières prédictions
    except Exception as e:
        click.echo(f'Erreur pendant la prédiction : {str(e)}')

@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="Fichier de données d'entraînement")
@click.option("--model_dump_filename", default="models/model.json", help="Fichier avec le modèle sauvegardé")
def evaluate(input_filename, model_dump_filename):
    """Évalue le modèle en utilisant la validation croisée."""
    df = make_dataset(input_filename)
    X, y = make_features(df)

    # Charger le modèle sauvegardé
    model = joblib.load(model_dump_filename)

    # Évaluer le modèle
    evaluate_model(model, X, y)

def evaluate_model(model, X, y):
    """Évalue le modèle en utilisant la validation croisée et retourne les métriques."""
    # Effectuer la validation croisée
    scores = cross_val_score(model, X, y, cv=5)  # Validation croisée à 5 plis
    click.echo(f'Scores de validation croisée : {scores}')
    click.echo(f'Moyenne des scores : {scores.mean()}')

    # Entraîner le modèle sur l'ensemble de données complet pour d'autres métriques d'évaluation
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculer des métriques supplémentaires
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    click.echo(f'Accuracy : {accuracy:.4f}')
    click.echo(f'Précision : {precision:.4f}')
    click.echo(f'Recall : {recall:.4f}')
    click.echo(f'Score F1 : {f1:.4f}')

cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
