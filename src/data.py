import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import os
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

input_file = RAW_DATA_DIR / 'dataset.csv'

def make_dataset(filename):
    """Load dataset from a CSV file."""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {filename} n'a pas été trouvé.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier {filename} est vide.")
        return None
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du fichier : {e}")
        return None


def split_and_save_dataset(dataset, output_dirname, test_size=0.2):
    """Split the dataset into train and test sets and save them as CSV files."""
    full_ds = make_dataset(dataset)
    if full_ds is None:
        return  # Ne rien faire si le chargement du dataset échoue

    train, test = train_test_split(full_ds, test_size=test_size)

    os.makedirs(output_dirname, exist_ok=True)  # Crée le dossier s'il n'existe pas

    train_file = f'{output_dirname}/train.csv'
    test_file = f'{output_dirname}/test.csv'

    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

    print(f'Dataset divisé : {len(train)} échantillons pour l\'entraînement et {len(test)} pour le test.')
    print(f'Données d\'entraînement enregistrées dans : {train_file}')
    print(f'Données de test enregistrées dans : {test_file}')


def download_dataset_nlp(input_name):
    """Download a dataset from a Google Sheets URL and save it as a CSV file."""
    url = 'https://docs.google.com/spreadsheets/d/1HBs08WE5DLcHEfS6MqTivbyYlRnajfSVnTiKxKVu7Vs/export?format=csv&gid=1482158622'
    response = requests.get(url)
    output_dir = os.path.dirname(input_name)
    if response.status_code == 200:
        with open(input_name, 'wb') as f:
            f.write(response.content)
        print(f"Fichier téléchargé avec succès et enregistré à {input_name}")
    else:
        print(f"Erreur lors du téléchargement du fichier: {response.status_code}")

