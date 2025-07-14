# IMPORT DEPENDENCY LIBS
import nltk
import praw
import pandas as pd
import time
import os
import nltk
from nltk.corpus import stopwords
import prawcore
from dotenv import load_dotenv
import os

# Loading .env variables
load_dotenv()


nltk.download('punkt_tab')

# --- DEFINE REDDIT API CREDENTIALS HERE ---
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = os.getenv("USER_AGENT")


# --- Setup Reddit API ---
reddit = praw.Reddit(
    client_id = CLIENT_ID,
    client_secret = CLIENT_SECRET,
    user_agent = USER_AGENT
)

# --- Ensure stopwords are available ---
nltk.download("stopwords")
nltk.download("punkt")

# --- Stopwords ---
# INSERT STOP WORDS IN THIS ARRAY TO REMOVE FROM SENTIMENT ANALYSIS
STOPWORDS = [
    "ang", "ng", "sa", "na", "mga", "at", "ito", "iyon", "si", "ni", "ay",
    "dahil", "pero", "kaya", "kung", "may", "wala", "ako", "ikaw", "kami",
    "kita", "siya", "nila", "natin", "kayo", "sila", "mo", "ko", "niya",
    "ayon", "lamang", "pa", "rin", "man", "nga", "tayo", "dito", "doon", 
    "ka", "mo", "hindi", "para", "yun", "yung", "lang",  "s", "talaga", "ma",
    "din", "naman", "ba", "mag", "eh", "kay", "nag", "sya"
]
filipino_stopwords = set(STOPWORDS)
english_stopwords = set(stopwords.words("english"))
all_stopwords = filipino_stopwords.union(english_stopwords)

# --- CONFIGURE THE VARIABLES ---
SUBREDDITS = ["Philippines", "pinoy", "Manila", "inthephilippines", "ChikaPH"]
CSV_FILENAME = "reddit_multi_subreddits_june.csv"
POSTS_PER_SUBREDDIT = 3000
POSTS_PER_BATCH = 50
SLEEP_BETWEEN_BATCHES = 2

# --- Clean and tokenize text ---
def clean_text(text):
    words = nltk.word_tokenize(text.lower())
    return " ".join([word for word in words if word.isalpha() and word not in all_stopwords])

# --- Save posts to CSV ---
def append_to_csv(posts_batch):
    if not posts_batch:
        print("⚠️  No posts to save in this batch.")
        return
    df = pd.DataFrame(posts_batch)
    if not os.path.exists(CSV_FILENAME):
        print(f"📄 Creating new file: {CSV_FILENAME}")
        df.to_csv(CSV_FILENAME, index=False)
    else:
        print(f"📄 Appending to existing file: {CSV_FILENAME}")
        df.to_csv(CSV_FILENAME, mode='a', index=False, header=False)

# --- Scrape from multiple subreddits ---
def fetch_posts_from_subreddits(subreddits):
    for sub in subreddits:
        print(f"🔍 Scraping from r/{sub}...")
        try:
            subreddit = reddit.subreddit(sub)
            all_posts = []
            count = 0

            for post in subreddit.top(limit=POSTS_PER_SUBREDDIT, time_filter="month"):
                all_posts.append({
                    "subreddit": sub,
                    "title": post.title,
                    "cleaned_title": clean_text(post.title),
                    "score": post.score,
                    "author": str(post.author),
                    "url": post.url,
                    "created": pd.to_datetime(post.created_utc, unit='s'),
                    "comments": post.num_comments
                })

                count += 1
                if count % POSTS_PER_BATCH == 0:
                    append_to_csv(all_posts)
                    print(f"✅ Appended {count} posts from r/{sub} so far...")
                    all_posts = []
                    time.sleep(SLEEP_BETWEEN_BATCHES)

            if all_posts:
                append_to_csv(all_posts)
                print(f"✅ Final batch of {len(all_posts)} posts from r/{sub} appended.\n")
            elif count == 0:
                print(f"⚠️  No posts found in r/{sub}. It may be empty or restricted.")

        except prawcore.exceptions.Forbidden:
            print(f"❌ Skipped r/{sub}: Forbidden (private/quarantined/banned)")
        except prawcore.exceptions.NotFound:
            print(f"❌ Skipped r/{sub}: Subreddit not found")
        except prawcore.exceptions.Redirect:
            print(f"❌ Skipped r/{sub}: Redirected (possibly banned)")
        except Exception as e:
            print(f"❌ Error with r/{sub}: {e}")
        time.sleep(SLEEP_BETWEEN_BATCHES)

# --- Start scraping ---
fetch_posts_from_subreddits(SUBREDDITS)