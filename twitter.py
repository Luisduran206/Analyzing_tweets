import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import click
import seaborn as sns
import nltk
import ssl
from wordcloud import WordCloud
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def csv_to_df_pandas(path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame.
    Parameters:
        path (str): The file path to the CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")

        return pd.DataFrame()


def transform_pandas_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract users, hashtags, links,and 'created' column to datetime if it exists.
    Parameters:
        df (pd.DataFrame): The input DataFrame to transform.
    Returns:
        pd.DataFrame: The transformed DataFrame with additional columns.
    """
    # Extract users, hashtags, and links from text using regex
    df["users"] = df["text"].apply(lambda x: re.findall(r"@(\w+)", x))
    df["hashtags"] = df["text"].apply(lambda x: re.findall(r"#(\w+)", x))
    df["links"] = df["text"].apply(lambda x: re.findall(r"(https?://\S+)", x))
    df["source"] = df["statusSource"].str.extract(r">(.*?)<")
    df.drop("statusSource", axis=1, inplace=True)
    df.drop("retweeted", axis=1, inplace=True)
    df.drop("favorited", axis=1, inplace=True)
    df.drop(
        ["longitude", "latitude", "replyToSN", "replyToSID", "replyToUID"],
        axis=1,
        inplace=True,
    )
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df["truncated"] = df["truncated"].astype(bool)
    df["isRetweet"] = df["isRetweet"].astype(bool)

    return df


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the sentiment of the text in the DataFrame.
    Parameters:
        df (pd.DataFrame): The input DataFrame with tweets.
    Returns:
        pd.DataFrame: The DataFrame with added columns for polarity and subjectivity.
    """
    df["polarity"] = df["text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["subjectivity"] = df["text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df


def cleaning_text(text: str) -> str:
    """
    Function that cleans a text for better analysis. Deletes links (form "httt..."), hashtag symbol,
    retweet word ("RT"), mentions (form "@...") and ":".

    Parameters:
        text (str): The text for cleaning.
    Returns:
        text (str): text after cleaning.
    """
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"^RT", "", text)
    text = re.sub(r"@", "", text)
    text = re.sub(r":", "", text)
    return text


def preprocess_text(text):
    """
    Function that tokenize a given text for better understanding and getting polarity.
    Uses NLTK.

    Parameters:
        text (str): The text you want to tokenize.
    Returns:
        text (str): text after tokenize, a cleaning version of text that leads to a
                    better calculation of polarity.
    """
    # Tokenize the text
    text = cleaning_text(text)
    tokens = word_tokenize(text.lower())

    # Remove stop words
    custom_stopwords = set(stopwords.words("english")).union({"u", "s", "rt"})

    filtered_tokens = [
        token for token in tokens if token not in custom_stopwords and len(token) > 2
    ]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = " ".join(lemmatized_tokens)
    return processed_text


def plot_most_used_token(df: pd.DataFrame, output_dir: str):
    """
    Generates and saves a word cloud image of the most frequently used tokens in a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing a column 'text'
                           with text data to be tokenized and analyzed.
        output_dir (str): Directory where the word cloud image will be saved.
    Returns:
        None
    """
    df_tokens = pd.DataFrame({"tokens": df["text"].apply(preprocess_text)})
    df_tokens_exploded = df_tokens["tokens"].str.split().explode()

    token_counts = df_tokens_exploded.value_counts()
    token_counts_dict = token_counts.head(50).to_dict()

    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(token_counts_dict)

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "most_common_word_cloud.png"))
    plt.close()


def plot_analyze_sentiment_by_source(df: pd.DataFrame, output_dir: str):
    """
    Plots and saves a boxplot visualizing the distribution of sentiment polarity
    grouped by the source of tweets.

    Parameters:
        df (pd.DataFrame): DataFrame containing tweet data with columns 'statusSource' and sentiment analysis results.
        output_dir (str): Directory where the resulting plot image will be saved.
    Returns:
        None
    """
    sources_df = df["source"]
    sources_df.columns = ["source"]
    sources_df = sources_df.value_counts()
    sources_of_interest = sources_df.head(10).index.tolist()

    df_filtered = df[df["source"].isin(sources_of_interest)]
    df_sentiment_by_source = analyze_sentiment(df_filtered)

    # Configuration of the boxplot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Create a boxplot to visualize polarity grouped by source
    sns.boxplot(
        data=df_sentiment_by_source, x="source", y="polarity", palette="viridis"
    )

    # Configuration of title and labels
    plt.title("Distribución de polaridad de sentimiento por fuente de tweet")
    plt.xlabel("Fuente del tweet")
    plt.ylabel("Polaridad del sentimiento")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "analyze_sentiment_by_source_distribution.png")
    )
    plt.close()


def plot_top_hashtags(df: pd.DataFrame, output_dir: str):
    """
    Plot the frequency of the top 10 hashtags.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        :param df:
        :param output_dir:
    """
    hashtags = df["hashtags"].explode().value_counts().head(10)

    plt.figure(figsize=(10, 5))

    hashtags.plot(kind="bar", color="skyblue")

    plt.title("Top 10 Hashtags")
    plt.xlabel("Hashtags")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_hashtags.png"))
    plt.close()


def most_mentioned_users(df, output_dir: str):
    """
    Function to find and plot the most mentioned users in a column of tweets.

    Parameters:
    - df: DataFrame containing the tweets.
    -output_dir (str): Directory where the resulting plot image will be saved.
    Returns:
    - None.
    """
    top_n = 10

    # Extract mentions (users) as lists by tweet
    df["mentions"] = df["text"].str.findall(r"@(\w+)")

    # Expand the list of mentions into a single column
    expanded_mentions = df["mentions"].explode()

    # Count the frequency of each mentioned user
    mentions_counts = expanded_mentions.value_counts().reset_index()
    mentions_counts.columns = ["User", "Frequency"]

    # Filter the top most mentioned users (top_n)
    top_mentions = mentions_counts.head(top_n)

    # Plot the results
    top_mentions.plot(
        kind="bar",
        x="User",
        y="Frequency",
        legend=False,
        color="skyblue",
        figsize=(10, 6),
    )
    plt.xlabel("User")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} most mentioned users")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout
    plt.savefig(os.path.join(output_dir, "most_mentioned_users.png"))
    plt.close()


def plot_polarity_stats(df: pd.DataFrame, output_dir: str):
    """
    Plots the mean and median polarity as a bar chart.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the polarity data.
        output_dir (str): The directory to save the plot.
    """
    df["polarity"] = df["text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["subjectivity"] = df["text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    # Plot the distribution of polarity and subjectivity in a histogram with a red line showing the distribution mean
    plt.figure(figsize=(20, 10))

    # Histogram of polarity
    plt.subplot(1, 2, 1)
    sns.histplot(df["polarity"], kde=True, bins=50)
    plt.axvline(df["polarity"].mean(), color="red")
    plt.title("Distribución de polaridad")
    plt.xlabel("Polaridad")
    plt.ylabel("Cantidad de tweets")

    # Histogram of subjectivity
    plt.subplot(1, 2, 2)
    sns.histplot(df["subjectivity"], kde=True, bins=50)
    plt.axvline(df["subjectivity"].mean(), color="red")
    plt.title("Distribución de subjetividad")
    plt.xlabel("Subjetividad")
    plt.ylabel("Cantidad de tweets")
    plt.legend(["Media"])

    plt.savefig(os.path.join(output_dir, "polarity_mean_median.png"))
    plt.close()


def plot_pie_top_hastag(df: pd.DataFrame, output_dir: str):
    mentioned_hashtags = df["text"].str.extractall(r"#(\w+)")
    mentioned_hashtags.columns = ["hashtags"]

    # Count the occurrences of each hashtag
    hashtag_counts = mentioned_hashtags["hashtags"].explode().value_counts()

    # Filter out hashtags with less than 250 occurrences
    filtered_hashtags = hashtag_counts[hashtag_counts >= 250]

    # Plot the distribution of hashtags in a bar and a pie chart
    plt.figure(figsize=(20, 10))

    # Bar chart
    plt.subplot(1, 2, 1)
    filtered_hashtags.plot(kind="bar")
    plt.title("Distribución de hashtags")
    plt.xlabel("Hashtags")
    plt.ylabel("Cantidad de apariciones")

    # Pie chart
    plt.subplot(1, 2, 2)
    filtered_hashtags.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Distribución de hashtags (porcentaje)")
    plt.ylabel("")

    plt.savefig(os.path.join(output_dir, "pie_top_hastag.png"))
    plt.close()


def write_markdown_report(input_files, output_dir="markdown"):
    """
    Function to write a Markdown report from a list of input Markdown files.
    Parameters:
        input_files: List of input Markdown files to combine.
        output_dir: Directory where the output Markdown file will be saved.
    Returns:
        None.
    """
    combined_content = []

    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            combined_content.append(content)
            combined_content.append("\n\n")

    with open(
        os.path.join(output_dir, "analisis_final.md"), "w", encoding="utf-8"
    ) as f:
        f.write("".join(combined_content))


@click.command()
@click.option(
    "--path",
    prompt="\t> Path to CSV file (./data/avengers_endgame.csv)",
    help="Path to the CSV file to analyze.",
)
def main(path):
    """
    Main function
    Parameters:
        path (str): The file path to the CSV file to analyze.
    """

    if not os.path.exists("pandoc"):
        os.mkdir("pandoc")

    # STEP 0: Load data from CSV file
    df = csv_to_df_pandas(path)

    # If df is empty, exit the program, because the file not exist or path incorrect
    if df.empty:
        return

    # STEP 1: Transform the DataFrame
    print("\t1-Transformando dataframe ...")
    df = transform_pandas_df(df)

    # STEP 2: Perform sentiment analysis
    print("\t2-Analizando sentimientos ...")
    df = analyze_sentiment(df)
    print("\t3-Creando graficas a partir de los siguientes datos... \n")
    plot_polarity_stats(df, output_dir="pandoc")

    # STEP 3: Display basic statistical information about the DataFrame
    print(df.describe())

    # STEP 4: Plot the extracted data and save it
    plot_top_hashtags(df, output_dir="pandoc")  # Call the function to plot hashtags

    plot_analyze_sentiment_by_source(
        df, output_dir="pandoc"
    )  # Call the function to plot Analysis of sentiment per source

    plot_most_used_token(
        df, output_dir="pandoc"
    )  # Call the function to plot most used tokens

    most_mentioned_users(df, output_dir="pandoc")

    plot_pie_top_hastag(df, output_dir="pandoc")

    print("\t Graficas creadas correctamente")

    # STEP 5: Generate a Markdown
    # print("\t4-Generando Markdown ...")
    input_files = [
        "markdown/introduccion.md",
        "markdown/metodologia.md",
        "markdown/analisis.md",
    ]
    # write_markdown_report(input_files)  # Write markdown report
    # print(
    #    "\n> Fichero Markdown generado correctamente en: markdown/analisis_final.md \n"
    # )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    main()
# ./data/avengers_endgame.csv
