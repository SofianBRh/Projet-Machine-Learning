Projet-Machine-Learning : Librairie Python pour le Nettoyage de Données avec Pandas

Nettoyage de données l'aide de Pandas. 

    Importation de données :
        Prend en charge les formats JSON, CSV, ou des dictionnaires Python en tant qu'input.

    Stockage dans des Dataframes :
        Permet de stocker les données dans des dataframes pour une manipulation aisée.


Fonctions utiles pour le nettoyage de données :

Charger un CSV vers un Dataframe :

    df = pd.read_csv("data.csv")
    print(df)
    => prend un fichier CSV en entrée
    => créer un dataframe à partir des données CSV

Récupérer les infos du Dataframe (données vides, noms des colonnes, etc.) :
    df.info()
    => fonction qui s'applique à un dataframe
    => renvoie les infos du dataframe
    
Sommer le nombre de valeurs nulles dans chaque colonne :
    print(df.isnull().sum())
    => prend un dataframe en entrée
    => permet de compter le nombre d'éléments nuls dans chaque colonne

Vérifier s'il y a des données vides dans le Dataset :
    print(df.isnull().values.any())
    => renvoie True ou False

Compte le nombre de valeurs manquantes totales :
    print(df.isnull().sum().sum())
    => renvoie un nombre

Visualiser les données en doublons :
    data.loc[data['nom_colonne'].duplicated(keep=False),:]
    => prend un colonne en entrée
    => renvoie les éléments doublons 
    
