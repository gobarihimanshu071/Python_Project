
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn

def load_and_clean(file_path):
    data=pd.read_csv(file_path,na_values=["NA"])
    print("Data size:", data.shape)
    print("Columns: ",data.columns.tolist())

    data.dropna(subset=["ANNUAL"],inplace=True)
    print("Rows after dropping missing Annual: ", len(data))

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for  month in months:
        data[month]=data.groupby("SUBDIVISION")[month].transform(lambda x: x.fillna(x.median()))

    data["High_Rainfall"]=(data["ANNUAL"]>data["ANNUAL"].quantile(0.75)).astype(int)
    return data   

def plot_yearly_trend(data,area):
    subset = data[data["SUBDIVISION"]== area]
    plt.figure(figsize=(10,5))
    plt.plot(subset["YEAR"], subset["ANNUAL"], color="blue")
    plt.title("Rainfall over time- {area}")
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.grid(True)
    plt.show()

def plot_monthly_spread(data):
    months= data[["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]]
    plt.figure(figsize=(12,6))
    sns.boxplot(data=months)
    plt.title("How rainfall varies by mobnth")
    plt.xlabel("Month")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.show()

def train_rain_model(data):
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    X=data[months]
    y=data["High_Rainfall"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2 , random_state=42)

    model=RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)
    from sklearn.metrics import accuracy_score
    score=accuracy_score(y_test,y_pred)
    print(f"Model score: {score:.2f}")

    return model, X_test

def plot_what_matters(model, months):
    importance = model.feature_importances_
    plt.figure(figsize=(10,6))
    sns.barplot(x=importance, y=months)
    plt.title("Which Months Matter Most")
    plt.xlabel("Importance")
    plt.ylabel("Month")
    plt.show()



def main():
    print("Project starting...")
    file_path = r"c:\Users\ASUS\Downloads\Sub_Division_IMD_2017.csv"
    df=load_and_clean(file_path)
    print("Data preview")
    print(df.head())

    plot_yearly_trend(df,"Andaman & Nicobar Islands")
    plot_monthly_spread(df)

    model, X_test = train_rain_model(df)

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    plot_what_matters(model, months)

if __name__ == "__main__":
    main()
