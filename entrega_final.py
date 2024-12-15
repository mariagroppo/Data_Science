import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('./CO2+Emissions/visualizing_global_co2_data.csv')
# print(df.shape) # (50598, 79)

# ----- MANIPULACION DE DATOS ----------------------------------------------------------------
new_df = df[df['iso_code'].notna()] # Uso paises y no continentes. (42142, 79)
new_df.info()
#print(new_df.isna().sum()) # Suma de valores nulos por columna
#print((new_df.isna().sum()/new_df.shape[0])*100) # % de valores nulos por columna

# Elimino columnas con alta cantidad de nulos.
columnas_usar = ['country', 
                 'year', 
                 'population',
                 'cement_co2',
                 'gas_co2',
                 'coal_co2',
                 'consumption_co2',
                 'flaring_co2',
                 'land_use_change_co2',
                 'oil_co2',
                 'other_industry_co2']
df_filtrado = new_df[columnas_usar]
print((df_filtrado.isna().sum()/df_filtrado.shape[0])*100) # % de valores nulos por columna


# Reemplazo valores nulos con el promedio de la columna del país correspondiente
for col in ['cement_co2',
                 'gas_co2',
                 'coal_co2',
                 'consumption_co2',
                 'flaring_co2',
                 'land_use_change_co2',
                 'oil_co2',
                 'other_industry_co2']:
    # Calculo el promedio por país para la columna actual
    promedio_pais = df_filtrado.groupby('country')[col].transform('mean')
    # Reemplazo los valores nulos con el promedio correspondiente
    df_filtrado.loc[:, col] = df_filtrado[col].fillna(promedio_pais)

print((df_filtrado.isna().sum()/df_filtrado.shape[0])*100) # % de valores nulos por columna

#df_filtrado.dropna(inplace=True) # No me sirve, ya que pierdo mas del 50% de los datos.

# Para columnas con % menores a 10% de nulos los reemplazo por 0
df_filtrado.loc[:, 'cement_co2'] = df_filtrado['cement_co2'].fillna(0)
df_filtrado.loc[:, 'gas_co2'] = df_filtrado['gas_co2'].fillna(0)
df_filtrado.loc[:, 'coal_co2'] = df_filtrado['coal_co2'].fillna(0)
df_filtrado.loc[:, 'flaring_co2'] = df_filtrado['flaring_co2'].fillna(0)
df_filtrado.loc[:, 'land_use_change_co2'] = df_filtrado['land_use_change_co2'].fillna(0)
df_filtrado.loc[:, 'oil_co2'] = df_filtrado['oil_co2'].fillna(0)

print((df_filtrado.isna().sum()/df_filtrado.shape[0])*100) # % de valores nulos por columna

# Elimino las columnas consumption_co2 y other_industry_co2
columnas_usar = ['country', 
                 'year', 
                 'population',
                 'cement_co2',
                 'gas_co2',
                 'coal_co2',
                 'flaring_co2',
                 'land_use_change_co2',
                 'oil_co2']
final_df = df_filtrado[columnas_usar]
print((final_df.isna().sum()/final_df.shape[0])*100) # % de valores nulos por columna


print(final_df.shape)
final_df.dropna(inplace=True) # Se borra el 12% de los datos para que todas las columnas me queden con valores no nulos.
print(final_df.shape)
print((final_df.isna().sum()/final_df.shape[0])*100) # % de valores nulos por columna


print(final_df.country.unique())
final_df['emisiones_totales'] = final_df[['cement_co2', 'gas_co2', 'coal_co2', 'flaring_co2', 'land_use_change_co2', 'oil_co2']].sum(axis=1) # Calculo las emisiones totales de CO₂ por país y año


# Necesito crear una variable categorica: Nivel de Industrialización
# Clasifico los países según el nivel de emisiones de CO₂ por sectores industriales en comparación con sus emisiones totales. 
final_df['indus_level'] = final_df['emisiones_totales'].apply(
    lambda x: 'Alta industrialización' if x > 200 else 'Media industrialización' if x > 50 else 'Baja industrialización'
)

# Verificar los resultados
category_counts = final_df['indus_level'].value_counts()

# ------------------ MODELO -----------------------------------------------------------------------------------
X = final_df[['cement_co2', 'gas_co2', 'coal_co2', 'flaring_co2', 'land_use_change_co2', 'oil_co2','emisiones_totales']]
y = final_df[['indus_level']]

y.value_counts(normalize=True)*100 # Para ver el balance de los datos. Por debajo del 80% no deberia tener inconvenientes.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Encoders: cambio de variables a numeros ----------------
rules = {'Alta industrialización': 0, 'Media industrialización':1, 'Baja industrialización':2}
y_train['indus_level'] = y_train['indus_level'].map(rules)
y_test['indus_level'] = y_test['indus_level'].map(rules)


# ENTRENAMIENTO DEL MODELO -----------------------------------------
clf = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train,y_train)

train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

print(classification_report(y_train, train_pred)) 
print(classification_report(y_test, test_pred))

# -----------------------------------
scaler = MinMaxScaler() # Inicializo

# Creo conjuntos de datos
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

train_pred = knn.predict(X_train_scaled)
test_pred = knn.predict(X_test_scaled)

print(classification_report(y_train, train_pred))
print(classification_report(y_test, test_pred))