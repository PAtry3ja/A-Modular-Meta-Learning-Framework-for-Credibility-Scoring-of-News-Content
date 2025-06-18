import pandas as pd
import re
import string
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

#functions --- 
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def load_combined_data():
    true_df = pd.read_csv("data/True.csv")
    fake_df = pd.read_csv("data/Fake.csv")

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    return df.sample(frac=1).reset_index(drop=True)
#------

df = load_combined_data()

df["full_text"] = df["title"] + " " + df["text"]
df["clean_text"] = df["full_text"].apply(clean_text)

#save---
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned_data.csv", index=False)
