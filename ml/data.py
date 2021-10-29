import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class config:
    columns = ['BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
       'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime',
       'StockOptionLevel', 'WorkLifeBalance']

def read_data(path):
    data = pd.read_csv(path)
    data["Attrition"] = data["Attrition"].map({'Yes':1, 'No':0})
    constant_cols = data.nunique()[data.nunique() == 1].keys().tolist()
    data.drop(constant_cols, axis=1, inplace=True)
    cols = [c for c in data.columns if c in config().columns+["Attrition"]]

    return data[cols]

def replace_categories(df, var, target):
    
    ordered_labels = df.groupby([var])[target].mean().to_frame().sort_values(target).index
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    df[var] = df[var].map(ordinal_label)
    
    return ordinal_label


def preprocessing(data):

    target = "Attrition"
    cat_cols = [
        cat for cat in data.select_dtypes("O").columns.tolist()]

    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    
    label_list =[]
    for var in cat_cols:
        lbl = replace_categories(df, var, target)
        label_list.append((var, lbl)) 

    return df, dict(label_list)



def data_split(data):

    target = "Attrition"
    train_cols = [c for c in data.columns
                  if c not in [target]]

    x_train, x_test, y_train, y_test = train_test_split(data[train_cols],
                                                        data[target],
                                                        test_size=0.2,
                                                        random_state=2021,
                                                        stratify=data[target])

    return x_train, x_test, y_train, y_test, train_cols, target

