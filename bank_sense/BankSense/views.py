import base64
import re
import os
from datetime import datetime
from collections import Counter
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from matplotlib import pyplot as plt
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from .models import Review, VisualiData, ServiceModel
from django.http import JsonResponse
import io
from google.cloud import storage
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords, wordnet
import torch
from tqdm import tqdm

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Initialize the summarizer pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# View to list all reviews
def dashboard_view(request):
    service_name = request.GET.get('service', None)

    bank_name = request.GET.get('bank', 'CIBC')

    try:
        df = read_csv_from_gcs("text-mining-labeled-data", "final_labeled_reviews")
        service_list = read_services_from_gcs("text-mining-labeled-data", "filtered_keywords.csv")

        visuali_data = analyze_service_sentiment(df, bank_name, service_name)
        # visuali_data.positive_reviews = summarize_reviews(visuali_data.positive_reviews)
        # visuali_data.negative_reviews = summarize_reviews(visuali_data.negative_reviews)

        service_list.remove('Keyword')  # removing the header

        # Refine text for positive and negative word clouds
        positive_text = " ".join(visuali_data.positive_reviews)
        negative_text = " ".join(visuali_data.negative_reviews)

        # Generate refined word clouds
        positive_wordcloud = generate_word_cloud(positive_text, sentiment='positive')
        negative_wordcloud = generate_word_cloud(negative_text, sentiment='negative')

        return render(request, 'BankSense/index.html', {
            'visuali_data': visuali_data,
            'service_list': service_list,
            'positive_wordcloud': positive_wordcloud,
            'negative_wordcloud': negative_wordcloud
        })

    except Exception as e:
        print(str(e))
        return render(request, 'BankSense/index.html', {'error': str(e)})


stop_words = set(stopwords.words("english"))


# Helper function to generate a refined word cloud
def generate_wordcloud(text, sentiment):
    # Define stopwords to remove common and unhelpful words
    common_stopwords = {"the", "and", "to", "in", "it", "is", "this", "that", "with", "for", "on", "as", "was",
                        "are", "but", "be", "have", "at", "or", "from", "app", "bank", "service", "customer", "one",
                        "like", "can", "get", "use", "using", "also", "would", "will", "make", "good", "bad"}

    # Tokenize text and filter out stopwords and short words
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in common_stopwords and len(word) > 2]

    # Generate the word cloud
    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color="white",
        colormap='Greens' if sentiment == 'positive' else 'Reds',  # Green for positive, red for negative
        max_words=50
    ).generate(" ".join(filtered_words))

    # Convert the word cloud to a base64 image
    buffer = BytesIO()
    plt.figure(figsize=(4, 2))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')


def analyze_service_sentiment(df, bank_name="ALL"):
    visuali_data = VisualiData()

    # Filter by bank name if provided
    if bank_name != "ALL":
        df = df[df["bank"] == bank_name]

    # Aggregate data for analysis
    aggregated_data = (
        df.groupby(["bank"])
        .agg(
            total_reviews=("sentiment_score", "size"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_count=("predicted_sentiment", lambda x: (x == "positive").sum()),
            neutral_count=("predicted_sentiment", lambda x: (x == "neutral").sum()),
            negative_count=("predicted_sentiment", lambda x: (x == "negative").sum())
        )
        .nlargest(5, "total_reviews")
        .reset_index()
    )

    # Populate VisualiData instance
    for _, row in aggregated_data.iterrows():
        if bank_name == "ALL" or row["bank"] == bank_name:
            visuali_data.bank_name = row["bank"]
            visuali_data.total_reviews = row["total_reviews"]
            visuali_data.avg_rating = row["avg_sentiment"]
            visuali_data.pos_count = row["positive_count"]
            visuali_data.neu_count = row["neutral_count"]
            visuali_data.neg_count = row["negative_count"]

            # Optionally populate positive and negative reviews, if available in df
            visuali_data.positive_reviews = df[(df["bank"] == row["bank"]) & (df["predicted_sentiment"] == "positive")][
                "review_text"].tolist()
            visuali_data.negative_reviews = df[(df["bank"] == row["bank"]) & (df["predicted_sentiment"] == "negative")][
                "review_text"].tolist()

    return visuali_data


def analyze_service_sentiment(df, bank_name, service_name=None):
    keywords_to_avoid = ['app', 'interface', 'ui', 'layout', 'design', 'update', 'fingertips', 'bug', 'fingerprint',
                         'version']
    common_st_services = ['Credit', 'Security', 'Online banking', 'Mortgage', 'Fee']

    visualidata = VisualiData()
    visualidata.bank_name = bank_name

    if service_name is not None:
        # common_st_services.insert(0, service_name.replace('-', ' '))  # add selected service at front
        service_name = ' '.join(word.replace('-', ' ') for word in service_name.split())
        common_st_services = [service_name]
        visualidata.searched_st_service = service_name
        # common_st_services.pop() # remove last one.

    # assigning top 5 services and 1 selected service (if selected)
    for service in common_st_services:
        servicemodel = ServiceModel()
        servicemodel.name = service
        visualidata.common_services.append(servicemodel)

    # Filter reviews by the given bank name and check for the service name in the review
    bank_reviews = df[df['bank'] == bank_name]
    visualidata.bank_name = bank_name
    visualidata.total_reviews = len(bank_reviews)

    # generating 5 common service related data from filtered dataframe
    for service in visualidata.common_services:
        for _, row in bank_reviews.iterrows():
            review = row['review_text']
            review = str(review).lower()

            if service.name.lower() in review:
                sentiment = row['predicted_sentiment']

                # Increment or decrement sentiment_counter based on the sentiment
                if sentiment == 'positive':
                    service.pos_count += 1

                    if len(visualidata.positive_reviews) < 5:
                        visualidata.positive_reviews.append(review)
                elif sentiment == 'negative':
                    service.neg_count += 1
                    if len(visualidata.negative_reviews) < 5:
                        visualidata.negative_reviews.append(review)
                elif sentiment == 'neutral':
                    service.neu_count += 1

    # generating bank related data
    for _, row in bank_reviews.iterrows():
        sentiment = row['predicted_sentiment']
        if sentiment == 'positive':
            visualidata.pos_count += 1
        elif sentiment == 'negative':
            visualidata.neg_count += 1
        elif sentiment == 'neutral':
            visualidata.neu_count += 1

    common_banks = ['RBC', 'Scotiabank', 'CIBC', 'NBC', 'TD', 'BMO']
    common_banks.remove(bank_name)  # removing searched bank
    visualidata.curr_bank_list = common_banks

    if service_name is not None:
        # initializing count to zero

        for bank in common_banks:
            visualidata.service_at_other_banks[bank] = 0

        for _, row in df.iterrows():
            review = row['review_text']
            review = str(review).lower()

            if service_name.lower() in review:
                sentiment = row['predicted_sentiment']
                bank = row['bank']
                try:
                    if sentiment == 'positive':
                        curr_count = visualidata.service_at_other_banks[bank]
                        visualidata.service_at_other_banks[bank] = curr_count + 1
                except Exception as e:
                    print(e)

    for key, value in visualidata.service_at_other_banks.items():
        print(f"{key}: {value}")

    return visualidata


############################################  utility functions  #######################################################


def read_csv_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    csv_data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(csv_data))
    return df


def read_services_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    csv_data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(csv_data), header=None)

    keywords_list = []
    for _, row in df.iterrows():
        keywords_list.append(row[0])

    return keywords_list


def get_wordnet_pos(word):
    """Map POS tag to first character accepted by lemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_text(text, stop_words):
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return " ".join(lemmatized_tokens)


def generate_word_cloud(text, sentiment):
    # Ensure stopwords are loaded
    stop_words = set(stopwords.words('english'))

    # Define stopwords to remove common and unhelpful words
    custom_stopwords = {"the", "and", "to", "in", "it", "is", "this", "that", "with", "for", "on", "as", "was",
                        "are", "but", "be", "have", "at", "or", "from", "app", "bank", "service", "customer", "one",
                        "like", "can", "get", "use", "using", "also", "would", "will", "make", "good", "bad", "app",
                        "bank", "service", "customer", "one", "like", "can", "get", "use", "using",
                        "also", "would", "will", "make", "still", "even"}
    stop_words.update(custom_stopwords)
    text = preprocess_text(text, stop_words)

    # Tokenize text and filter out stopwords and short words
    words = re.findall(r'\b\w+\b', text.lower())
    sentiment_words = []
    # Initialize progress bar for word processing
    print("Processing words for sentiment analysis:")
    for word in tqdm(words, desc="Analyzing", unit="word"):
        if word not in stop_words and len(word) > 2:
            # Analyze sentiment of each word using RoBERTa
            inputs = tokenizer(word, return_tensors="pt")
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            positive_score, neutral_score, negative_score = scores[2], scores[1], scores[0]

            # Filter words based on sentiment
            if sentiment == 'positive' and positive_score > 0.4:
                sentiment_words.append(word)
            elif sentiment == 'negative' and negative_score > 0.3:
                sentiment_words.append(word)

    # Generate the word cloud
    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color="white",
        colormap='Greens' if sentiment == 'positive' else 'Reds',  # Green for positive, red for negative
        max_words=50
    ).generate(" ".join(sentiment_words))

    # Convert the word cloud to a base64 image
    buffer = BytesIO()
    plt.figure(figsize=(4, 2))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')


def overall_bank_sentiment_dashboard(request):
    #print("nltk path: ",nltk.data.path)
    df = read_csv_from_gcs("text-mining-labeled-data", "final_labeled_reviews")
    # Step 1: Map sentiment strings to numerical values
    df["sentiment_score"] = df["predicted_sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})

    aggregated_data = (
        df.groupby(["bank"])
        .agg(
            total_reviews=("sentiment_score", "size"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_count=("predicted_sentiment", lambda x: (x == "positive").sum()),
            neutral_count=("predicted_sentiment", lambda x: (x == "neutral").sum()),
            negative_count=("predicted_sentiment", lambda x: (x == "negative").sum())
        )
        .nlargest(5, "total_reviews")
        .reset_index()
    )

    top_positive_reviews = (
        df[df["predicted_sentiment"] == "positive"]
        .sort_values(by="rating", ascending=False)
        .head(3)  # Adjust as needed for more or fewer reviews
    )
    top_negative_reviews = (
        df[df["predicted_sentiment"] == "negative"]
        .sort_values(by="rating", ascending=True)
        .head(3)  # Adjust as needed for more or fewer reviews
    )
    top_positive_reviews_text = " ".join(top_positive_reviews["review_text"])
    top_negative_reviews_text = " ".join(top_negative_reviews["review_text"])
    #top_positive_reviews_text = summarize_large_text(top_positive_reviews_text)
    #top_negative_reviews_text = summarize_large_text(top_positive_reviews_text)
    # positive_wordcloud = generate_word_cloud(top_positive_reviews_text,sentiment='positive')
    # negative_wordcloud = generate_word_cloud(top_negative_reviews_text,sentiment='negative')

    # Convert the aggregated data to a dictionary format for the template
    aggregated_data_json = aggregated_data.to_dict(orient="records")
    service_list = read_services_from_gcs("text-mining-labeled-data", "filtered_keywords.csv")

    # get all bank names
    service_list = read_services_from_gcs("text-mining-labeled-data", "filtered_keywords.csv")
    service_list.remove('Keyword')
    #return render(request, 'BankSense/index.html', {'visuali_data': visuali_data, 'service_list': service_list})

    # Pass the data as context to the template
    context = {
        "aggregated_data": aggregated_data_json,
        "top_positive_reviews": top_positive_reviews_text,
        "top_negative_reviews": top_negative_reviews_text,
        "service_list": service_list,
        # "positive_wordcloud": positive_wordcloud,
        # "negative_wordcloud": negative_wordcloud,
    }
    return render(request, 'BankSense/index_temp.html', context)


def summarize_reviews(reviews):
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Iterate over each review in the list
    for i in range(len(reviews)):
        if len(reviews[i]) > 150:
            summary = summarizer(reviews[i], max_length=250, min_length=30, do_sample=False)
            reviews[i] = summary[0]['summary_text']

    return reviews


def extract_sentiment_keywords(text, threshold=0.5):
    positive_keywords = []
    negative_keywords = []

    # Tokenize the text and obtain embeddings
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    scores = torch.softmax(logits, dim=1).detach().numpy()

    # Check each token's contribution to sentiment
    for i, token in enumerate(tokens):
        word = tokenizer.convert_ids_to_tokens([inputs.input_ids[0][i]])[0]
        # Assign tokens based on sentiment threshold
        if scores[0][2] > threshold:  # Positive sentiment score
            positive_keywords.append(word)
        elif scores[0][0] > threshold:  # Negative sentiment score
            negative_keywords.append(word)

    return set(positive_keywords), set(negative_keywords)
