from math import log
from collections import Counter
import pandas as pd


def calc_raw_prob_inplace(label: dict, tot_words_in_class: int, unique: int):
    for element in label:
        label[element] = (label[element] + 1) / (tot_words_in_class + unique)  # freq


# apply bayes theorem for predicting
def predict_raw(
    text: str,
    real: dict,
    fake: dict,
    prob_real: float,
    tot_words_real: int,
    tot_words_fake: int,
    n_unique: int,
) -> bool:
    # prob(real|words)=prob(words|real)*prob(real)/prob(words)
    prob_r = log(prob_real)
    prob_f = log(1 - prob_real)
    text = text.lower()
    list_of_words = text.split()
    for word in list_of_words:
        if word in real:
            prob_r += log(real[word])
        else:  # laplace smoothing
            prob_r += log(1 / (tot_words_real + n_unique))
        if word in fake:
            prob_f += log(fake[word])
        else:
            prob_f += log(1 / (tot_words_fake + n_unique))

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
    text: str, idf_real: Counter, idf_fake: Counter, prob_real: float
) -> bool:
    prob_r = log(prob_real)
    prob_f = log(1 - prob_real)
    epsilon = 1e-9
    tf = Counter(text.split())
    total = tf.total()
    sum_idf_real = idf_real.total()
    sum_idf_fake = idf_fake.total()
    for term in tf.keys():
        prob_r += (log(tf[term] / total) + log((idf_real[term] + epsilon) / sum_idf_real))
        prob_f += (log(tf[term] / total) + log((idf_fake[term] + epsilon) / sum_idf_fake))

    return prob_r > prob_f
