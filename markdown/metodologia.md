# Metodología

## Datos

Utilizamos el dataset de los tuits de Avengers Endgame, avengers_endgame.csv.

La investigación está basada en el análisis contextual y especifico de los datos más relevantes obtenidos del archivo, a los cuales se ha llegado tras un proceso de valoración y elección de estos mismos.

Los resultados están en la sección Resultados.

## Código

Primeramente necesitaremos importr el siguiente conjunto de librerias:

```python
    from  collections  import  Counter
    import  re
    import  os
    import  pandas  as  pd
    import  matplotlib.pyplot  as  plt
    from  textblob  import  TextBlob
    import  click
    import  seaborn  as  sns
    import  nltk
    from  nltk.sentiment.vader  import  SentimentIntensityAnalyzer
    from  nltk.corpus  import  stopwords
    from  nltk.tokenize  import  word_tokenize
    from  nltk.stem  import  WordNetLemmatizer
```

Utilizamos la librería _pandas_ para la transformación del CSV al dataframe, y posteriormente para extraer datos derivados de los tuits, como la polaridad de los mismos.

También es resaltable el uso de _matplotlib_ para generar gráficos: histogramas, diagramas de barras, gráficos de dispersión de manera sencilla.

Remarcamos el uso de _nltk_ y _textblob_, el cual es muy importante para el análisis del sentimiento.

```python
    def csv_to_df_pandas(path: str) -> pd.DataFrame:
```

Esta función carga un archivo csv y lo convierte en un dataframe de _pandas_. Si el archivo no se encuentra, devuelve un dataframe vacío.

```python
    def transform_pandas_df(df: pd.DataFrame) -> pd.DataFrame:
```

Esta función transforma el dataframe extrayendo usuarios, hashtags y enlaces de la columna texto.

```python
    def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
```

Esta función analiza el sentimiento de cada texto en el dataframe usando _textblob_, agregando las columnas de polaridad y subjetividad.

```python
    def cleaning_text(text: str) -> str:
```

Esta función limpia el texto eliminando enlaces, hashtags, menciones y RTs para preparar los datos para un mejor análisis.

```python
    def preprocess_text(text):
```

Tokeniza el texto y elimina palabras utilizando _nltk_, asimismo lematiza para dejar las palabras en su raíz.

```python
    def plot_analyze_sentiment_by_source(df: pd.DataFrame, output_dir: str):
```

Crea un boxplot para visualizar la polaridad de los tuits basada en la fuente de los mismos, como puede ser: Android, iPhone, web; entre otros.

```python
    def plot_most_used_token(df: pd.DataFrame, output_dir: str):
```

Genera y guarda una tabla de los 50 tokens más usados en los textos.

```python
    def plot_top_hashtags(df: pd.DataFrame, output_dir: str):
```

Genera un gráfico de barras con los 10 hashtags más frecuentes en los textos.

```python
    def plot_sentiment_distribution(df: pd.DataFrame, output_dir: str):
```

Crea un histograma de la distribución de polaridad de sentimientos, en función de la frecuencia de los niveles de polaridad.

```python
    def main(path):
```

Es la función principal y el punto de entrada del script, realiza las siguientes funciones:

1. Cargar los datos del archivo CSV.
2. Transformar el dataframe.
3. Realizar el análisis de sentimientos.
4. Mostrar estadísticas básicas.
5. Generar las visualizaciones y guardarlas como imágenes.

## Ejecución

Para ejecutar el programa se crean automáticamente las carpetas, donde estén todas las imágenes y los datos, en: pandoc y data.
