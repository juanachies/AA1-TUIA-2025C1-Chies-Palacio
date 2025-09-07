import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   # u otros scalers
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------------

### carga datos de dataset en dataframe
file_path= 'uber_fares.csv'

df = pd.read_csv(file_path) 

# -----------------------------------

### visualizacion de algunos datos
df.head()

# ----------------------------------

### Columnas, ¿cuáles son variables numéricas y cuales variables categóricas?
df.columns

# ----------------------------------

#Primero verificamos las columnas que contienen valores nulos
df.info()

df.isnull().sum()

df[df.isnull().any(axis=1)] 

df.shape
df = df.dropna()
df.shape

# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='fare_amount'), df['fare_amount'], test_size=0.2, random_state=42)

# ----------------------------------

X_train.describe()
X_train.info()

y_train.info()

print(y_train)
# ----------------------------------

sns.boxplot(X_train)

# ----------------------------------