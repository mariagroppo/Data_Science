import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./CO2+Emissions/visualizing_global_co2_data.csv')
# print(df.shape) # (50598, 79)

# ----- MANIPULACION DE DATOS ----------------------------------------------------------------
cantidad_inicial_paises = len(df['country'].unique()) # Analisis de paises listados
#print(df.info())

# Analisis de paises con iso_code nulo.
df_paises_iso_code_nulo = df[df['iso_code'].isna()] # Filtro filas para saber cuales son los paises que tiene iso_code nulo.
df_paises_iso_code_NO_nulo = df[df['iso_code'].notna()] # Filtro filas para saber cuales son los paises que tiene iso_code nulo.
df_paises_iso_code_nulo_agrupado = df_paises_iso_code_nulo.groupby('country').size()
df_paises_iso_code_NO_nulo_agrupado = df_paises_iso_code_NO_nulo.groupby('country').size()

# Analisis de nulos cuando filtro unicamente los datos de paises (no regiones)
new_df = df[df['iso_code'].notna()]
#print(new_df.isna().sum())

# Analisis de paises con population nulo.
df_paises_population_nulo = df[df['population'].isna()] # Filtro filas para saber cuales son los paises que tiene population nulo.
#print(df_paises_population_nulo)

df_filtered_only_countries = df[(df['country'].notna()) & (df['population'] > 0)]
#print(df_filtered)
new_df = df_filtered_only_countries[df_filtered_only_countries['iso_code'].notna()]

# -------------------------------------------------------------------------------------------------------------------------------
# 1.	¿Qué país emite la mayor cantidad de CO2 por año? -----------------------------------------------------------------------
new_df['emisiones_totales'] = new_df[['cement_co2', 'gas_co2', 'coal_co2', 'consumption_co2', 'flaring_co2', 'land_use_change_co2', 'oil_co2', 'other_industry_co2']].sum(axis=1) # Calculo las emisiones totales de CO₂ por país y año
df_totales = new_df.groupby(['country', 'year'], as_index=False)['emisiones_totales'].sum() # Agrupo por pais y ano para obtener el total de emisiones
df_max_emisiones = df_totales.loc[df_totales.groupby('year')['emisiones_totales'].idxmax()] # Agrupo por ano y obtengo el país con la mayor cantidad de emisiones
#print(df_max_emisiones.tail(20))

# China = Matplotlib
df_china = new_df[new_df['country'] == 'China']

plt.figure(figsize=(10, 6))
plt.plot(df_china['year'], df_china['emisiones_totales'], label='Emisiones Totales', color='black', linewidth=2)
plt.plot(df_china['year'], df_china['cement_co2'], label='Cemento', linestyle='--', color='blue')
plt.plot(df_china['year'], df_china['gas_co2'], label='Gas', linestyle='--', color='orange')
plt.plot(df_china['year'], df_china['coal_co2'], label='Carbon', linestyle='--', color='green')
plt.plot(df_china['year'], df_china['consumption_co2'], label='Consumo', linestyle='--', color='pink')
plt.plot(df_china['year'], df_china['flaring_co2'], label='Quema', linestyle='--', color='red')
plt.plot(df_china['year'], df_china['land_use_change_co2'], label='Uso de la tierra', linestyle='--', color='violet')
plt.plot(df_china['year'], df_china['oil_co2'], label='Petroleo', linestyle='--', color='grey')
plt.plot(df_china['year'], df_china['other_industry_co2'], label='Otras industrias', linestyle='--', color='yellow')

plt.xlabel('Año')
plt.ylabel('Emisiones de CO₂')
plt.title('Emisiones de CO₂ en China por Origen y Totales')
plt.legend()
plt.grid(True)
plt.show()


#United States + Seaborn
df_usa = new_df[new_df['country'] == 'United States']

plt.figure(figsize=(10, 6))
sns.lineplot(x='year', y='emisiones_totales', data=df_usa, label='Emisiones Totales', color='black', linewidth=2)
sns.lineplot(x='year', y='cement_co2', data=df_usa, label='Cemento', linestyle='--', color='blue')
sns.lineplot(x='year', y='gas_co2', data=df_usa, label='Gas', linestyle='--', color='orange')
sns.lineplot(x='year', y='coal_co2', data=df_usa, label='Carbon', linestyle='--', color='green')

plt.xlabel('Año')
plt.ylabel('Emisiones de CO₂')
plt.title('Emisiones de CO₂ en Estados Unidos por Origen (Cemento, Gas, Carbon) y Totales')
plt.legend()
plt.grid(True)
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------
# 2.	¿Como es la evolución de la emisión de CO2 por persona por año?
df_anio = new_df.groupby('year', as_index=False).agg({'population': 'sum', 'emisiones_totales': 'sum'})

# Calcular las emisiones totales per cápita
df_anio['emisiones_per_capita'] = df_anio['emisiones_totales'] / df_anio['population']
plt.figure(figsize=(10, 6))
plt.plot(df_anio['year'], df_anio['emisiones_per_capita'], color='blue', marker='o', linestyle='-')
plt.xlabel('Año')
plt.ylabel('Emisiones de CO₂ per cápita')
plt.title('Evolución de las Emisiones Totales de CO₂ per Cápita por Año')
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------
# 3.	¿Cuál es la relación entre el PIB per cápita y las emisiones de CO2?
new_df['PIB_per_capita'] = new_df['gdp'] / new_df['population']

correlacion = new_df[['PIB_per_capita', 'emisiones_totales']].corr()
print("Correlación entre PIB per cápita y emisiones totales de CO₂:")
print(correlacion)

plt.figure(figsize=(10, 6))
plt.scatter(new_df['PIB_per_capita'], new_df['emisiones_totales'], alpha=0.5, color='blue')
plt.xlabel('PIB per cápita')
plt.ylabel('Emisiones Totales de CO₂')
plt.title('Relación entre PIB per cápita y Emisiones de CO₂')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='PIB_per_capita', y='emisiones_totales', data=new_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel('PIB per cápita')
plt.ylabel('Emisiones Totales de CO₂')
plt.title('Relación entre PIB per cápita y Emisiones de CO₂ con Regresión Lineal')
plt.grid(True)
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------
# 4.	¿Qué países han logrado reducir sus emisiones de CO2 en relación con su crecimiento económico?

df_china['intensidad_carbono'] = df_china['emisiones_totales'] / df_china['gdp']
df_usa['intensidad_carbono'] = df_usa['emisiones_totales'] / df_usa['gdp']

plt.figure(figsize=(12, 6))
plt.plot(df_china['year'], df_china['intensidad_carbono'], label='China', color='red')
plt.xlabel('Año')
plt.ylabel('Intensidad de Carbono (Emisiones CO₂ / PBI)')
plt.title('Evolución de la Intensidad de Carbono - China')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_usa['year'], df_usa['intensidad_carbono'], label='EEUU', color='blue')
plt.xlabel('Año')
plt.ylabel('Intensidad de Carbono (Emisiones CO₂ / PBI)')
plt.title('Evolución de la Intensidad de Carbono - EEUU')
plt.legend()
plt.grid()
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------
# ¿Cómo varían las emisiones de CO₂ de diferentes fuentes en EEUU a lo largo de los años?

df_melted = pd.melt(df_usa, id_vars=['country'], value_vars=['cement_co2', 'gas_co2', 'coal_co2', 'consumption_co2', 'flaring_co2', 'land_use_change_co2', 'oil_co2', 'other_industry_co2'],
                    var_name='Tipo de Emisión', value_name='Emisiones Totales')

# Crear el boxplot para cada tipo de emisión
plt.figure(figsize=(10, 6))
df_melted.boxplot(column='Emisiones Totales', by='Tipo de Emisión', grid=False)
plt.title('Variabilidad de Emisiones de CO₂ por Tipo de Origen')
plt.suptitle('')
plt.xlabel('Tipo de Origen de Emisión')
plt.ylabel('Emisiones Totales de CO₂')
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------
# Histograma de emisiones totales
plt.figure(figsize=(10, 5))
plt.hist(df_usa['emisiones_totales'], bins=15, color='skyblue', edgecolor='black', density=True)
plt.title('Distribución de Emisiones Totales de CO₂')
plt.xlabel('Emisiones Totales de CO₂')
plt.ylabel('Frecuencia')
plt.show()



