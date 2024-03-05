import pandas as pd

# Charger le fichier CSV dans un DataFrame
df = pd.read_csv('movies.csv')

# Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)

# Supprimer les doublons
df.drop_duplicates(inplace=True)

# Réinitialiser les index après la suppression
df.reset_index(drop=True, inplace=True)

# Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
df.to_csv('donnees_nettoyees.csv', index=False)

print("Le nettoyage des données est terminé. Les données nettoyées ont été enregistrées dans 'donnees_nettoyees.csv'.")
