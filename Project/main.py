import pandas as pd
from preprocess import clean_text, tokenize
from bayes import calc_raw_prob_inplace, predict_raw, predict_tf_idf, IDF
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


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

        # precomputing idf hash tables
        self.idf_real = IDF(self.train_true)
        self.idf_fake = IDF(self.train_false)

    def NaiveBayes(self):
        real = tokenize(self.train_data, 1)
        fake = tokenize(self.train_data, 0)
        n_unique = len(real.keys() | fake.keys())
        tot_words_real = sum(real.values())
        tot_words_fake = sum(fake.values())
        calc_raw_prob_inplace(real, tot_words_real, n_unique)
        calc_raw_prob_inplace(fake, tot_words_fake, n_unique)

        # Make predictions
        self.test_data["predicted_label_raw"] = self.test_data["text"].apply(
            lambda x: predict_raw(
                x, real, fake, self.prob_real, tot_words_real, tot_words_fake, n_unique
            )
        )


        real_labels = self.test_data["label"]
        predicted_labels = self.test_data["predicted_label_raw"]

        # Generate confusion matrix
        conf_matrix = confusion_matrix(real_labels, predicted_labels)

        # Calculate metrics
        accuracy = accuracy_score(real_labels, predicted_labels)
        precision = precision_score(real_labels, predicted_labels)
        recall = recall_score(real_labels, predicted_labels)
        f = f1_score(real_labels, predicted_labels)

        # Create confusion matrix heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",cbar=False)

        # Display metrics on the plot window
        plt.text(2, -0.3, f'    Accuracy: {accuracy:.2f}', fontsize=12, color='black')
        plt.text(2, -0.1, f'    Precision: {precision:.2f}', fontsize=12, color='black')
        plt.text(2, 0.1, f'    Recall: {recall:.2f}', fontsize=12, color='black')
        plt.text(2, 0.3, f'    F-Score: {f:.2f}', fontsize=12, color='black')

        # Add plot labels and title
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def NaiveBayesTFIDF(self):
        self.test_data["predicted_label_tfidf"] = self.test_data["text"].apply(
            lambda x: predict_tf_idf(x, self.idf_real, self.idf_fake, self.prob_real)
        )

        real_labels = self.test_data["label"]
        predicted_labels = self.test_data["predicted_label_tfidf"]

        # Generate confusion matrix
        conf_matrix = confusion_matrix(real_labels, predicted_labels)

        # Calculate metrics
        accuracy = accuracy_score(real_labels, predicted_labels)
        precision = precision_score(real_labels, predicted_labels)
        recall = recall_score(real_labels, predicted_labels)
        f = f1_score(real_labels, predicted_labels)

        # Create confusion matrix heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",cbar=False)

        # Display metrics on the plot window
        plt.text(2, -0.3, f'    Accuracy: {accuracy:.2f}', fontsize=12, color='black')
        plt.text(2, -0.1, f'    Precision: {precision:.2f}', fontsize=12, color='black')
        plt.text(2, 0.1, f'    Recall: {recall:.2f}', fontsize=12, color='black')
        plt.text(2, 0.3, f'    F-Score: {f:.2f}', fontsize=12, color='black')

        # Add plot labels and title
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # Show the plot
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = NB(["dataset/train.tsv", "dataset/test.tsv"])
    model.NaiveBayesTFIDF()
