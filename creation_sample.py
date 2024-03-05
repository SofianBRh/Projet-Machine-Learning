import pandas as pd




if __name__ == "__main__":
    file=input("fichier csv à prélever : ")
    df=pd.read_csv(file)

    print(df.columns)

    echantillon=input("taille de l'échantillon : ")

    to_extract=df.head(int(echantillon))

    to_extract.to_csv("echantillon.csv",index=False)