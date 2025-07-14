import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# DEFINE SOURCE CSV
SOURCE_CSV = "reddit_multi_subreddits_june.csv"
RESULT_CSV = "results_sentiment_analysis.csv"
FILTERED_RESULT_CSV = "filtered_sentiment_analysis.csv"

# Download VADER lexicon (run once)
nltk.download('vader_lexicon')

# Load your CSV - change filename as needed
df = pd.read_csv(SOURCE_CSV)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if isinstance(text, str):
        return sia.polarity_scores(text)
    else:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis
df['sentiment'] = df['title'].apply(analyze_sentiment)
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df['pos'] = df['sentiment'].apply(lambda x: x['pos'])
df['neg'] = df['sentiment'].apply(lambda x: x['neg'])
df['sentiment_category'] = df['compound'].apply(categorize_sentiment)

# Optionally drop the dict column
df = df.drop(columns=['sentiment'])

# Show results
print(df.head())

# Save to new CSV
df.to_csv("reddit_multi_subreddits_june_sentiment.csv", index=False)

# --- Load the CSV with sentiment results ---
CSV_FILENAME = "reddit_multi_subreddits_june_sentiment.csv"
df = pd.read_csv(CSV_FILENAME)

# --- Define keywords to filter by ---
SA_KEYWORDS = ["ncap"]  # ADD WORDS TO THE ARRAY TO ANALYZE

# --- Filter rows containing any of the keywords in the cleaned title ---
filtered_df = df[df["cleaned_title"].str.contains('|'.join(SA_KEYWORDS), case=False, na=False)]

# --- Classify sentiment as Positive or Negative ---
def classify_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment classification
filtered_df["sentiment_category"] = filtered_df["compound"].apply(classify_sentiment)

# --- View filtered results ---
print(filtered_df[["cleaned_title", "sentiment_category", "compound"]].head(10))

# Optional: Save to CSV
filtered_df.to_csv(FILTERED_RESULT_CSV, index=False)

average_compound = filtered_df['compound'].mean()
print(average_compound)

print("\n-----------")
print("\nSENTIMENT ANALYSIS for the ff keywords:")
for word in SA_KEYWORDS:
    print ("\t `" + word + "`")

print("Average Compound: " + str(average_compound))

if average_compound >= 0.05:
    print("Is Generally POSITIVE")
elif average_compound <= -0.05:
    print("Is Generally NEGATIVE")
else:
    print("Is Generally NEUTRAL")
