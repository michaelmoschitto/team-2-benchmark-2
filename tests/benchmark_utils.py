import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error


def preprocess_video_games_df(path_to_video_games_csv):
    """
    This function will preprocesses the video games DF and returns the entire
    X and y dataframes.

    Input:
      path_to_video_games_csv (string): Path to csv
    Output:
      (X, y): Tuple of dataframes
        -> X : Dataframe with 5 columns of String Type
        -> y : Series with 1 column of float64 type
    """

    df = pd.read_csv(path_to_video_games_csv)
    df = df.drop(
        [
            "NA_Sales",
            "EU_Sales",
            "JP_Sales",
            "Other_Sales",
            "Critic_Score",
            "Critic_Count",
            "User_Score",
            "User_Count",
            "Developer",
            "Rating",
        ],
        axis=1,
    )

    # Set the year to categorical
    df["Year_of_Release"] = df["Year_of_Release"].apply(str)

    # Get rid of na rows
    df.dropna(inplace=True)

    # Set target variable
    y = df.pop("Global_Sales")
    return df, y


def preprocess_life_df(path_to_life_csv):
    """
    This function will preprocesses the life expectancy DF and returns the
    X and y dataframes.

    Input:
    path_to_life_csv (string): Path to csv
    Output:
    (X, y): Tuple of dataframes
        -> X : Dataframe with 7 columns of String Type
        -> y : Series with 1 column of float64 type
    """
    df = pd.read_csv(path_to_life_csv)

    # get y attrib
    y = df["Life expectancy "]

    # Type fixing
    df["Country"] = df["Country"].astype(str)
    df["Year"] = df["Year"].apply(str)
    df["Status"] = df["Status"].astype(str)

    # convert numerics to ranges
    per1000_bins = [i for i in range(0, 1001, 100)]
    per1000_labels = ["({i}-{j}]".format(i=i, j=i + 100) for i in per1000_bins[:-1]]
    per100_bins = [i for i in range(0, 101, 10)]
    per100_labels = ["({i}-{j}]".format(i=i, j=i + 10) for i in per100_bins[:-1]]
    per1_bins = [round(x * 0.1, 1) for x in range(0, 11)]
    per1_labels = ["({i}-{j}]".format(i=i, j=round(i + 0.1, 1)) for i in per1_bins[:-1]]

    df["Adult Mortality"] = pd.cut(
        df["Adult Mortality"], bins=per1000_bins, labels=per1000_labels
    ).astype(str)
    df["Hepatitis B %immun"] = pd.cut(
        df["Hepatitis B"], bins=per100_bins, labels=per100_labels
    ).astype(str)
    df["BMI"] = pd.cut(df[" BMI "], bins=per100_bins, labels=per100_labels).astype(str)
    df["Polio %immun"] = pd.cut(
        df["Polio"], bins=per100_bins, labels=per100_labels
    ).astype(str)
    df["Diphtheria %immun"] = pd.cut(
        df["Diphtheria "], bins=per100_bins, labels=per100_labels
    ).astype(str)
    df["Income composition of resources"] = pd.cut(
        df["Income composition of resources"], bins=per1_bins, labels=per1_labels
    ).astype(str)

    # selected features
    features = [
        "Country",
        "Year",
        "Status",
        "Adult Mortality",
        "Hepatitis B %immun",
        "BMI",
        "Polio %immun",
        "Diphtheria %immun",
        "Income composition of resources",
        "Life expectancy ",
    ]

    # only keeping main features
    df = df.loc[:, features]

    # drop missing values
    df = df.dropna()
    y_feature = "Life expectancy "
    y = df[y_feature]
    df = df.drop(y_feature, axis=1)
    return df, y


def preprocess_flare_df(**kwargs):
    df = pd.read_csv(kwargs["link"], sep=kwargs["sep"], names=kwargs["names"])

    # drop this because there is no variance
    df = df.drop(columns=["largest-spotarea"])

    # drop missing values
    df = df.dropna()

    # get y attrib
    y = df.pop("C")

    return df, y


def preprocess_titanic_df(path_to_titanic_csv):

    titanic_df = pd.read_csv(path_to_titanic_csv)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    titanic_df2 = titanic_df.loc[:, features]
    titanic_df2["CabinLetter"] = titanic_df2["Cabin"].str.slice(0, 1)
    X = titanic_df2.drop("Cabin", axis=1)
    X["CabinLetter"] = X["CabinLetter"].fillna("?")
    X["Pclass"] = X["Pclass"].astype(str)
    X["SibSp"] = X["SibSp"].astype(str)
    X["Parch"] = X["Parch"].astype(str)
    X["Age"] = (
        ((X["Age"].fillna(X["Age"].mean()) / 10).astype(int) * 10)
        .astype(int)
        .astype(str)
    )

    X = X.dropna()

    X2 = X.drop(columns="Fare")
    t = X["Fare"]

    return X2, t
