import pandas as pd
import pandas as pd
from nltk.corpus import stopwords
import re


stopW = set(stopwords.words("english"))  # set of stopwords

def clean_text(text: str) -> str:
    # Remove special quotes
    text = re.sub(r"[“”‘’]", "", text)
    
    # Remove commas and periods, while ensuring spaces are handled
    text = re.sub(r"[,.]", " ", text)  # Replace commas and periods with spaces

    # Remove possessive forms
    text = re.sub(r"'s\b", "", text)

    # Remove non-word characters (except for apostrophes)
    text = re.sub(r"[^\w\s']", "", text)

    text = re.sub(r"_+", " ", text) 
    # Convert to lowercase
    text = text.lower()
    
    # Split text into words and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopW]
    
    # Join the remaining words back into a string
    cleaned_text = " ".join(words)
    
    return cleaned_text


# word counts
def tokenize(data: pd.DataFrame, label: bool) -> dict:
    tokens = {}
    for text in data[data["label"] == label]["text"]:
        for word in str(text).split():
            if tokens.get(word) == None:
                tokens[word] = 1
            else:
                tokens[word] += 1
    return tokens