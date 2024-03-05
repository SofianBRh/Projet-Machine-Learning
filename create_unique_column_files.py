import pandas as pd




if __name__ == "__main__":
    file=input("fichier : ")
    df=pd.read_csv(file)

    for col in df.columns:
        print(col)
        unique_col=df[col].drop_duplicates()
        unique_col.to_csv("col_"+col+".csv", index=False)

