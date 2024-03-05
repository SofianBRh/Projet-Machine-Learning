import pandas as pd

#lis le fichier CSV
df = pd.read_csv('movies.csv')

# compte le nombre de lignes dans les données
nbr_lignes_avant = len(df)

# suppr des lignes avec des valeurs manquantes
df.dropna(inplace=True)

# suppr des lignes qui ont les mêmes valeurs pour les colonnes movies et year
df_clean = df.drop_duplicates(subset=['MOVIES', 'YEAR'])

#compte le nombre de lignes dans les données après nettoyage et affiche dans la console
nbr_lignes_apres = len(df)
print('Nombre de lignes avant nettoyage :', nbr_lignes_avant)
print('Nombre de lignes après nettoyage :', nbr_lignes_apres)