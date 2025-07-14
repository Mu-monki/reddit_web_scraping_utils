import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download("stopwords")
nltk.download("punkt")

# --- Load Data ---
CSV_FILENAME = "reddit_multi_subreddits_june.csv"
df = pd.read_csv(CSV_FILENAME)

# --- Combine Stopwords ---
STOPWORDS = [
    "ang", "ng", "sa", "na", "mga", "at", "ito", "iyon", "si", "ni", "ay",
    "dahil", "pero", "kaya", "kung", "may", "wala", "ako", "ikaw", "kami",
    "kita", "siya", "nila", "natin", "kayo", "sila", "mo", "ko", "niya",
    "ayon", "lamang", "pa", "rin", "man", "nga", "tayo", "dito", "doon", 
    "ka", "mo", "hindi", "para", "yun", "yung", "lang",  "s", "talaga", "ma",
    "din", "naman", "ba", "mag", "eh", "kay", "nag", "sya", "nung", "nasa",
    "nang", "nito", "eto", "philippine", "po", "nalang"
]
filipino_stopwords = set(STOPWORDS)
english_stopwords = set(stopwords.words("english"))
all_stopwords = filipino_stopwords.union(english_stopwords)

# --- Generate Word Cloud ---
text = " ".join(df["cleaned_title"].dropna().astype(str))
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    stopwords=all_stopwords,
    colormap="viridis"
).generate(text)

# --- Display ---
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Reddit Post Titles", fontsize=20)
plt.show()
