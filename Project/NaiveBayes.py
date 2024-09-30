import pandas as pd
from nltk.corpus import stopwords
import re
from collections import Counter
from math import log
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

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
def tokenize(data: pd.DataFrame, label: bool) -> Counter:
    tokens = Counter()
    for text in data[data["label"] == label]["text"]:
        tokens.update(str(text).split())
    return tokens


# apply bayes theorem for predicting
def predict_raw(
    text: str,
    real: Counter,
    fake: Counter,
    prob_real: float,
    tot_words_real: int,
    tot_words_fake: int,
    n_unique: int,
) -> bool:
    # prob(real|words)=prob(words|real)*prob(real)/prob(words)
    prob_r = log(prob_real)
    prob_f = log(1 - prob_real)
    list_of_words = text.split()
    for word in list_of_words:
        # laplace smoothing
        prob_r += log((1 + real[word]) / (tot_words_real + n_unique))
        prob_f += log((1 + fake[word]) / (tot_words_fake + n_unique))

    return prob_r > prob_f


def IDF(df: pd.DataFrame) -> Counter:
    idf = Counter()
    num_docs = len(df)  # Total number of documents

    for txt in df["text"]:
        # Split the text into words and use a set to count each word only once per document
        words_in_doc = set(txt.split())
        # This will count each word only once per document
        for word in words_in_doc:
            idf[word] += 1

    for word in idf:
        idf[word] = log(num_docs / (1 + idf[word])) + 1  # idf smooth

    return idf


def predict_tf_idf(
    text: str,
    real: Counter,
    fake: Counter,
    tot_words_real: int,
    tot_words_fake: int,
    n_unique: int,
    idf_real: Counter,
    idf_fake: Counter,
    prob_real: float,
) -> bool:
    prob_r = log(prob_real)
    prob_f = log(1 - prob_real)
    epsilon = 1e-9
    tf = Counter(text.split())
    total = tf.total()

    for term in tf.keys():
        prob_r += log(
            (tf[term] / total)
            * (idf_real[term] + epsilon)
            * ((real[term] + 1) / (tot_words_real + n_unique))
        )
        prob_f += log(
            (tf[term] / total)
            * (idf_fake[term] + epsilon)
            * ((fake[term] + 1) / (tot_words_fake + n_unique))
        )

    return prob_r > prob_f




class NB:
    def __init__(self, dataset_path):
        # Load and preprocess training and testing data
        self.train_data = pd.read_csv(dataset_path[0], delimiter="\t")
        self.train_data = self.train_data.drop(columns=["title", "subject", "date"])
        self.train_data["text"] = self.train_data["text"].apply(clean_text)

        self.test_data = pd.read_csv(dataset_path[1], delimiter="\t")
        self.test_data = self.test_data.drop(columns=["title", "subject", "date"])
        self.test_data["text"] = self.test_data["text"].apply(clean_text)

        # these constants must be global
        self.train_true = self.train_data[self.train_data["label"] == 1]
        self.train_false = self.train_data[self.train_data["label"] == 0]

        self.prob_real = len(self.train_true) / len(self.train_data)

        self.real = tokenize(self.train_data, 1)
        self.fake = tokenize(self.train_data, 0)
        self.n_unique = len(self.real.keys() | self.fake.keys())
        self.tot_words_real = sum(self.real.values())
        self.tot_words_fake = sum(self.fake.values())

        # precomputing idf hash tables
        self.idf_real = IDF(self.train_true)
        self.idf_fake = IDF(self.train_false)

    def NaiveBayes(self):
        # Make predictions
        self.test_data["predicted_label_raw"] = self.test_data["text"].apply(
            lambda x: predict_raw(
                x,
                self.real,
                self.fake,
                self.prob_real,
                self.tot_words_real,
                self.tot_words_fake,
                self.n_unique,
            )
        )

        real_labels = self.test_data["label"]
        predicted_labels = self.test_data["predicted_label_raw"]

        print(classification_report(real_labels, predicted_labels))

        # Generate confusion matrix
        conf_matrix = confusion_matrix(real_labels, predicted_labels)

        # Create confusion matrix heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

        # Add plot labels and title
        plt.title("Confusion Matrix for Raw Naive Bayes")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # Show the plot
        plt.tight_layout()
        plt.show()

    def NaiveBayesTFIDF(self):
        self.test_data["predicted_label_tfidf"] = self.test_data["text"].apply(
            lambda x: predict_tf_idf(
                x,
                self.real,
                self.fake,
                self.tot_words_real,
                self.tot_words_fake,
                self.n_unique,
                self.idf_real,
                self.idf_fake,
                self.prob_real,
            )
        )

        real_labels = self.test_data["label"]
        predicted_labels = self.test_data["predicted_label_tfidf"]

        print(classification_report(real_labels, predicted_labels))

        # Generate confusion matrix
        conf_matrix = confusion_matrix(real_labels, predicted_labels)

        # Create confusion matrix heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

        # Add plot labels and title
        plt.title("Confusion Matrix for Naive Bayes with TF-IDF")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # Show the plot
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = NB(["dataset/train.tsv", "dataset/test.tsv"])
    model.NaiveBayes()
    model.NaiveBayesTFIDF()
