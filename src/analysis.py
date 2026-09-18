import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def cargar_y_explorar_datos(filepath):
    print("--- 1. EXPLORACIÓN INICIAL ---")
    df = pd.read_csv(filepath)
    # Limpiar espacios en los nombres de las columnas por si acaso
    df.columns = df.columns.str.strip()
    
    print(f"Número de pasajeros: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    print("\nTipos de datos:\n", df.dtypes)
    print("\nValores faltantes:\n", df.isnull().sum())
    print(f"\nRegistros duplicados: {df.duplicated().sum()}")
    print("\nEstadísticas descriptivas:\n", df.describe(include='all'))
    return df

def limpiar_y_transformar_datos(df):
    print("\n--- 2. LIMPIEZA Y TRANSFORMACIÓN ---")
    df_clean = df.copy()

    # Tratamiento seguro de nulos según existencia de columna
    if 'Age' in df_clean.columns:
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    
    if 'Embarked' in df_clean.columns:
        df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
    
    if 'Cabin' in df_clean.columns:
        df_clean['Cabin'] = df_clean['Cabin'].fillna('Desconocido')

    # Creación de variables (Mínimo 2)
    sibsp = df_clean['SibSp'] if 'SibSp' in df_clean.columns else 0
    parch = df_clean['Parch'] if 'Parch' in df_clean.columns else 0
    df_clean['FamilySize'] = sibsp + parch + 1
    
    if 'Age' in df_clean.columns:
        labels_edad = ['Niño', 'Joven', 'Adulto', 'Adulto Mayor']
        bins_edad = [0, 12, 25, 60, 120]
        df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=bins_edad, labels=labels_edad)

    df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)

    return df_clean

def realizar_analisis(df):
    print("\n--- 3. ANÁLISIS DE DATOS ---")
    
    if 'Survived' in df.columns:
        supervivencia_global = df['Survived'].mean() * 100
        print(f"1. Porcentaje global de supervivencia: {supervivencia_global:.2f}%")

        if 'Sex' in df.columns:
            print("\n2. Supervivencia por Sexo:\n", df.groupby('Sex')['Survived'].mean() * 100)
        if 'Pclass' in df.columns:
            print("\n3. Supervivencia por Clase de pasajero:\n", df.groupby('Pclass')['Survived'].mean() * 100)
        if 'AgeGroup' in df.columns:
            print("\n4. Supervivencia por Grupo de Edad:\n", df.groupby('AgeGroup', observed=False)['Survived'].mean() * 100)
        print("\n5. Supervivencia según Acompañamiento:\n", df.groupby('IsAlone')['Survived'].mean() * 100)

def generar_visualizaciones(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Gráfico 1
    if 'Pclass' in df.columns and 'Sex' in df.columns and 'Survived' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', errorbar=None, palette='Set2')
        plt.title('Tasa de Supervivencia por Clase y Género')
        plt.ylabel('Porcentaje de Supervivencia')
        plt.savefig(os.path.join(output_dir, 'supervivencia_clase_sexo.png'))
        plt.close()

    # Gráfico 2
    if 'AgeGroup' in df.columns and 'Survived' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x='AgeGroup', y='Survived', palette='coolwarm', errorbar=None)
        plt.title('Tasa de Supervivencia por Grupo de Edad')
        plt.ylabel('Porcentaje de Supervivencia')
        plt.savefig(os.path.join(output_dir, 'supervivencia_edad.png'))
        plt.close()

    # Gráfico 3
    if 'Fare' in df.columns and 'Survived' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Survived', y='Fare', palette='Pastel1')
        plt.yscale('log')
        plt.title('Relación entre Tarifa Pagada y Supervivencia')
        plt.xticks([0, 1], ['No Sobrevivió', 'Sobrevivió'])
        plt.savefig(os.path.join(output_dir, 'supervivencia_tarifa.png'))
        plt.close()

    print(f"\nVisualizaciones guardadas correctamente en: {output_dir}")

if __name__ == "__main__":
    PATH_DATA = "data/train.csv"
    PATH_OUTPUT = "outputs/resultados"
    
    data = cargar_y_explorar_datos(PATH_DATA)
    data_limpia = limpiar_y_transformar_datos(data)
    realizar_analisis(data_limpia)
    generar_visualizaciones(data_limpia, PATH_OUTPUT)
