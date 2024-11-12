# -*- coding: utf-8 -*-
"""SVM_Loader_Labeller.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1j1zXWNn5d0t2Do1EqgUfMa6b9LYXNoZR
"""

from google.colab import drive
drive.mount('/content/gdrive')

!ls /content/gdrive/MyDrive

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character accepted by lemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return lemmatized_tokens

#reviews_tokenized = reviews.apply(preprocess_text)

from google.cloud import storage
from google.oauth2 import service_account
import os
import csv
from datetime import datetime

def download_latest_blob_with_timestamp(bucket_name):
    """Downloads the latest blob from the bucket and uses its timestamp as the filename."""
    SERVICE_ACCOUNT_FILE = 'amplified-brook-416922-6a39d3e05104.json'
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)

    storage_client = storage.Client(project='amplified-brook-416922', credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    # Get all blobs and then sort them by time_created
    blobs = list(bucket.list_blobs())
    blobs.sort(key=lambda blob: blob.time_created, reverse=True)
    # Sort in descending order (newest first)

    latest_blob = blobs[0] if blobs else None
    print(latest_blob.name)
    if latest_blob:
        # Get the timestamp from the blob's metadata
        timestamp = latest_blob.time_created.strftime("%Y%m%d_%H%M%S")
        # Format the timestamp as desired (e.g., YYYYMMDD_HHMMSS)

        # Create the destination filename using the timestamp
        #destination_file_name = os.path.join(destination_folder, f"downloaded_file_{timestamp}.csv")
        destination_file_name = f"unlabeled_raw_data.csv"

        # Download the latest blob to the specified folder
        latest_blob.download_to_filename(destination_file_name)
        print(f"Downloaded latest blob: gs://{bucket_name}/{latest_blob.name} to {destination_file_name}")
    else:
        print("No blobs found in the bucket.")


# Example usage
bucket_name = 'text-mining-source-dump'  # Replace with your bucket name
#destination_folder = '/path/to/your/destination/folder'  # Replace with your desired folder path

download_latest_blob_with_timestamp(bucket_name)

from google.cloud import storage
from google.oauth2 import service_account
import os

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    SERVICE_ACCOUNT_FILE = 'amplified-brook-416922-6a39d3e05104.json'  # Replace if needed
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
    )

    storage_client = storage.Client(project='amplified-brook-416922', credentials=credentials)  # Replace project ID if needed
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(
        f"Blob {source_blob_name} downloaded to {destination_file_name}."
    )

# Example usage:
bucket_name = "text-mining-source-dump"  # Replace with your bucket name
source_blob_name = "aggregated_data.csv"  # Replace with the name of the file in the bucket
destination_file_name = "unlabeled_raw_data.csv"  # Replace with the desired local filename

download_blob(bucket_name, source_blob_name, destination_file_name)

def get_review_vector(review, model):
    review_vec = np.zeros(100)  # 100 is the vector size used in Word2Vec
    count = 0
    for word in review:
        if word in model.wv.key_to_index:  # check if word is in the vocabulary
            review_vec += model.wv[word]
            count += 1
    return review_vec / count if count > 0 else review_vec

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=3000)  # Use top 3000 words for features

import pandas as pd
import numpy as np
import joblib

# ... (Your existing functions: preprocess_text, get_review_vector) ...

# Load the saved SVM model
model_save_name = 'classifier.joblib'
path = F"/content/gdrive/My Drive/{model_save_name}"
svm_classifier = joblib.load(path)

# Load the new dataset
new_data = pd.read_csv('unlabeled_raw_data.csv')
new_reviews = new_data['review_text']

label_mapping = {1: 'positive', -1: 'negative', 0: 'neutral'}

# Preprocess the new reviews
new_reviews_tokenized = new_reviews.apply(preprocess_text)

word2vec_model = Word2Vec(sentences=new_reviews_tokenized, vector_size=100, window=5, min_count=2, sg=1)


# Convert to Word2Vec vectors
new_X_word2vec = np.array([get_review_vector(review, word2vec_model) for review in new_reviews_tokenized])

# Convert to TF-IDF vectors (assuming tfidf_vectorizer is already fitted)
new_reviews_joined = [' '.join(review) for review in new_reviews_tokenized]
X_tfidf = tfidf_vectorizer.fit_transform(new_reviews_joined).toarray()
X_combined = np.hstack((new_X_word2vec, X_tfidf))
new_X_tfidf = tfidf_vectorizer.transform(new_reviews_joined).toarray()
# Combine Word2Vec and TF-IDF features
new_X_combined = np.hstack((new_X_word2vec, new_X_tfidf))

# Predict the sentiment for each review using the loaded model
new_predictions = svm_classifier.predict(new_X_combined)
new_predictions_text = [label_mapping[pred] for pred in new_predictions]

# Print or store the predictions as needed
# for i, prediction in enumerate(new_predictions_text):
#     print(f"Review {i + 1}: {prediction}")

# Save the predictions in the original DataFrame
new_data['predicted_sentiment'] = new_predictions_text

# Save to a new CSV file
new_data.to_csv('labeled_review1.csv', index=False)

print("Predicted labels saved to labeled_reviews1.csv")

from google.cloud import storage
from google.oauth2 import service_account
import os

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    SERVICE_ACCOUNT_FILE = 'amplified-brook-416922-6a39d3e05104.json'  # Replace if needed
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE
    )

    storage_client = storage.Client(project='amplified-brook-416922', credentials=credentials)  # Replace project ID if needed
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(
        f"Blob {source_blob_name} downloaded to {destination_file_name}."
    )

# Example usage:
bucket_name = "text-mining-labeled-data"  # Replace with your bucket name
source_blob_name = "labeled_reviews1"  # Replace with the name of the file in the bucket
destination_file_name = "labeled_reviews.csv"  # Replace with the desired local filename

download_blob(bucket_name, source_blob_name, destination_file_name)

import pandas as pd

def concatenate_csv(file1, file2, output_file):
    """Concatenates two CSV files into a single CSV file.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        output_file (str): Path to the output CSV file.
    """

    # Read the CSV files into Pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Concatenate the DataFrames
    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)

    print(f"CSV files concatenated and saved to {output_file}")


# Example usage:
file1 = 'final_labeled_reviews.csv'
file2 = 'labeled_review1.csv'
output_file = 'concatenated_file.csv'

concatenate_csv(file1, file2, output_file)

from google.cloud import storage
from google.oauth2 import service_account
import os
import csv

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    #bucket_name = "text-mining-source-dump"
    # The path to your file to upload
    #source_file_name = "C:\Users\Subhram Satyajeet\OneDrive - University of Windsor\Desktop\Internship Project 2\Review_Dataset_google_apple_net_banking\Dataset\bmo_google_before_2016.csv"
    # The ID of your GCS object
    #destination_blob_name = "bmo_google_before_2016"

    #Setting the service account credentials for authentication
    SERVICE_ACCOUNT_FILE = 'amplified-brook-416922-6a39d3e05104.json' #key file name
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE) #setting credentials using key file

    storage_client = storage.Client(project='amplified-brook-416922' ,credentials = credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

upload_blob('text-mining-labeled-data','concatenated_file.csv','final_labeled_reviews')