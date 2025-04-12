import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import confusion_matrix
import sklearn
from xgboost import XGBClassifier
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.arima.model import ARIMA

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
    plt.title(f"Rainfall over time- {area}")
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.grid(True)
    plt.show()

def plot_monthly_spread(data):
    months= data[["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]]
    plt.figure(figsize=(12,6))
    sns.boxplot(data=months)
    plt.title("How rainfall varies by month")
    plt.xlabel("Month")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.show()

def train_rain_model(data):
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    X=data[months]
    y=data["High_Rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2 , random_state=42)

    rf_model=RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred=rf_model.predict(X_test)
    rf_accuracy=accuracy_score(y_test,rf_pred)
    rf_precision=precision_score(y_test,rf_pred)
    print("Random Forest Results")
    print(f"Accuracy: {rf_accuracy:.2f}")
    print(f"Precision: {rf_precision:.2f}")

    gb_model=GradientBoostingClassifier(n_estimators=100,random_state=42)
    gb_model.fit(X_train,y_train)
    gb_pred=gb_model.predict(X_test)
    gb_accuracy=accuracy_score(y_test,gb_pred)
    gb_precision=precision_score(y_test,gb_pred)

    print("Gradient Boosting Results")
    print(f"Accuracy: {gb_accuracy:.2f}")
    print(f"Precision: {gb_precision:.2f}")

    xgb_model=XGBClassifier(n_estimators=100,random_state=42,eval_metric='logloss')
    xgb_model.fit(X_train,y_train)
    xgb_pred=xgb_model.predict(X_test)
    xgb_accuracy=accuracy_score(y_test,xgb_pred)
    xgb_precision = precision_score(y_test,xgb_pred)
    print("XGBBoost Results:")
    print(f" Accuracy:{xgb_accuracy:.2f}")
    print(f" Precision:{xgb_precision:.2f}")

    return rf_model,gb_model,xgb_model, X_test, y_test

def plot_confusion_matrix(rf_model,gb_model,xgb_model, X_test, y_test):
    rf_pred= rf_model.predict(X_test)
    gb_pred= gb_model.predict(X_test)
    xgb_pred=xgb_model.predict(X_test)

    rf_cm=confusion_matrix(y_test, rf_pred)
    gb_cm=confusion_matrix(y_test, gb_pred)
    xgb_cm=confusion_matrix(y_test, xgb_pred)

    fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(18,6))
    sns.heatmap(rf_cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Low","High"],yticklabels=["Low","High"],ax=ax1)
    ax1.set_title("Random Forest Confusion Matrix", fontsize=12, pad=10)
    ax1.set_xlabel("Predicted", fontsize=10)
    ax1.set_ylabel("Actual", fontsize=10)

    sns.heatmap(gb_cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Low","High"],yticklabels=["Low","High"],ax=ax2)
    ax2.set_title("Gradient Boosting Confusion Matrix", fontsize=12, pad=10)
    ax2.set_xlabel("Predicted", fontsize=10)
    ax2.set_ylabel("Actual", fontsize=10)

    sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax3)
    ax3.set_title("XGBoost Confusion Matrix", fontsize=12, pad=10)
    ax3.set_xlabel("Predicted", fontsize=10)
    ax3.set_ylabel("Actual", fontsize=10)

    plt.tight_layout(pad=3.0)
    plt.show()

def plot_what_matters(rf_model,gb_model,xgb_model, months):
    rf_importance = rf_model.feature_importances_
    gb_importance = gb_model.feature_importances_
    xgb_importance = xgb_model.feature_importances_

    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,6))

    sns.barplot(x=rf_importance, y=months,ax=ax1)
    ax1.set_title("Random Forest Feature Importance", fontsize=12, pad=10)
    ax1.set_xlabel("Importance", fontsize=10)
    ax1.set_ylabel("Month", fontsize=10)
    ax1.tick_params(axis='y',labelsize=8)

    sns.barplot(x=gb_importance, y=months,ax=ax2)
    ax2.set_title("Gradient Boosting Feature Importance", fontsize=12, pad=10)
    ax2.set_xlabel("Importance", fontsize=10)
    ax2.set_ylabel("Month", fontsize=10)
    ax2.tick_params(axis='y',labelsize=8)

    sns.barplot(x=xgb_importance, y=months, ax=ax3)
    ax3.set_title("XGBoost Feature Importance", fontsize=12, pad=10)
    ax3.set_xlabel("Importance", fontsize=10)
    ax3.set_ylabel("Month", fontsize=10)
    ax3.tick_params(axis='y', labelsize=8)

    plt.tight_layout(pad=3.0)
    plt.show()
def plot_correlation_heatmap(data):
    monthly_data=data[["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]]
    corr_matrix=monthly_data.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f")
    plt.title("Correlation Heatmap of Monthly Rainfall")
    plt.show()

def plot_avg_rainfall_heatmap(data):
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    avg_rainfall = data.groupby("SUBDIVISION")[months].mean()
    fig,ax=plt.subplots(figsize=(18,14))
    sns.heatmap(avg_rainfall, cmap="YlGnBu", annot=False,ax=ax)
    ax.set_title("Average Monthly Rainfall by Subdivision",fontsize=12,pad=20)
    ax.set_xlabel("Month",fontsize=8,labelpad=10)
    ax.set_ylabel("Subdivision",fontsize=8,labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.25)
    plt.show()

def plot_3d_rainfall(data, area):
    subset = data[data["SUBDIVISION"] == area].copy()

    years = []
    months= []
    rainfall = []

    month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, 
                 "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    
    for _, row in subset.iterrows():
        year = row["YEAR"]
        for month_name, month_num in month_map.items():
            years.append(year)
            months.append(month_num)
            rainfall.append(row[month_name])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(years, months, rainfall, c=rainfall, cmap='viridis', s=50)

    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    ax.set_zlabel("Rainfall (mm)")
    ax.set_title(f"3D Rainfall Plot - {area}", fontsize=12, pad=20)

    ax.set_yticks(list(month_map.values()))
    ax.set_yticklabels(list(month_map.keys()))
    
    plt.colorbar(scatter, label="Rainfall (mm)")
    
    plt.show()


def forecast_annual_rainfall_arima(df, forecast_years=10):

    rainfall_series = df.groupby("YEAR")["ANNUAL"].mean()
    rainfall_series = rainfall_series.sort_index()
    rainfall_series.index = pd.Index(rainfall_series.index.astype(int))

    model = ARIMA(rainfall_series, order=(2, 1, 2))  
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_years)

    last_year = rainfall_series.index[-1]

    future_years = list(range(df["YEAR"].max() + 1, df["YEAR"].max() + forecast_years + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(rainfall_series.index, rainfall_series.values, label="Historical Rainfall", marker="o")
    plt.plot(future_years, forecast, label="Forecasted Rainfall", linestyle="dashed", marker="o", color="red")
    plt.xlabel("Year")
    plt.ylabel("Annual Rainfall (mm)")
    plt.title(f"Annual Rainfall Forecast (Next {forecast_years} Years)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nForecasted Rainfall:")
    for year, rain in zip(future_years, forecast):
        print(f"{year}: {rain:.2f} mm")


def main():
    print("Project starting...")
    file_path = r"c:\Users\ASUS\Downloads\Sub_Division_IMD_2017.csv"
    df=load_and_clean(file_path)
    print("Data preview")
    print(df.head())

    plot_yearly_trend(df,"Andaman & Nicobar Islands")
    plot_monthly_spread(df)
    plot_3d_rainfall(df, "Andaman & Nicobar Islands")

    rf_model,gb_model,xgb_model, X_test ,y_test= train_rain_model(df)
    plot_confusion_matrix(rf_model,gb_model,xgb_model,X_test,y_test)

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    plot_what_matters(rf_model,gb_model,xgb_model, months)

    plot_correlation_heatmap(df)
    plot_avg_rainfall_heatmap(df)

    forecast_annual_rainfall_arima(df, forecast_years=10)


if __name__ == "__main__":
    main()

