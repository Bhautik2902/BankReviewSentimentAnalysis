import base64
import json
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
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords, wordnet
# import torch
from tqdm import tqdm

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

from PIL import Image, ImageDraw, ImageFont
import io
import base64


def generate_no_data_image():
    try:
        # Create an empty image with white background
        img = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Add text "No Data Found"
        text = "No Data Found"

        # Attempt to load a font; fallback to default if unavailable
        try:
            font = ImageFont.truetype("arial.ttf", size=20)  # Ensure the font is installed
        except IOError:
            font = ImageFont.load_default()

        # Use textbbox instead of textsize
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]  # Right - Left
        text_height = bbox[3] - bbox[1]  # Bottom - Top

        # Calculate position to center the text
        position = ((400 - text_width) // 2, (200 - text_height) // 2)
        draw.text(position, text, fill=(0, 0, 0), font=font)

        # Save the image to a BytesIO buffer
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Convert the image buffer to a base64 string
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

        return img_base64

    except Exception as e:
        print(f"Error generating no data image: {e}")
        return None


# View to list all reviews
def dashboard_view(request):
    service_name = request.GET.get('service', None)
    bank_name = request.GET.get('bank', 'CIBC')
    try:
        json_data = read_json_from_gcs("text-mining-labeled-data", "json_database.json")
        service_list = read_services_from_gcs("text-mining-labeled-data", "filtered_keywords.csv")
        dict_object = fetch_data_by_bank_and_service(json_data, bank_name, service_name)

        if dict_object is not None:
            visuali_data = convert_dict_to_model(dict_object)
            positive_wordcloud = create_word_cloud_image(visuali_data.positive_word_list, 'positive')
            negative_wordcloud = create_word_cloud_image(visuali_data.negative_word_list, 'negative')

        else:

            if service_name not in service_list:
                return render(request, 'BankSense/index.html', {
                    'visuali_data': VisualiData(),
                    'service_list': service_list,
                    'positive_wordcloud': generate_no_data_image(),
                    'negative_wordcloud': generate_no_data_image()
                })

                # if precomputed data is unavailable utilize text-miniing-labeled-data raw file
            df = read_csv_from_gcs("text-mining-labeled-data", "final_labeled_reviews")
            visuali_data = analyze_service_sentiment(df, bank_name, service_name)

            # Refine text for positive and negative word clouds
            positive_text = " ".join(visuali_data.positive_reviews)
            negative_text = " ".join(visuali_data.negative_reviews)

            # Generate refined word clouds
            positive_wordcloud = generate_word_cloud(positive_text, sentiment='positive')
            negative_wordcloud = generate_word_cloud(negative_text, sentiment='negative')

            #To do: include the code to append  the resultant data to json_database

        return render(request, 'BankSense/index.html', {
            'visuali_data': visuali_data,
            'service_list': service_list,
            'positive_wordcloud': positive_wordcloud,
            'negative_wordcloud': negative_wordcloud
        })

    except Exception as e:
        print(str(e))
        return render(request, 'BankSense/index.html', {'error': str(e)})

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
            visualidata.pos_service_at_other_banks[bank] = 0
            visualidata.neg_service_at_other_banks[bank] = 0
            visualidata.neu_service_at_other_banks[bank] = 0

        for _, row in df.iterrows():
            review = row['review_text']
            review = str(review).lower()

            if service_name.lower() in review:
                sentiment = row['predicted_sentiment']
                bank = row['bank']
                try:
                    if sentiment == 'positive':
                        curr_count = visualidata.pos_service_at_other_banks[bank]
                        visualidata.pos_service_at_other_banks[bank] = curr_count + 1
                    elif sentiment == 'negative':
                        curr_count = visualidata.neg_service_at_other_banks[bank]
                        visualidata.neg_service_at_other_banks[bank] = curr_count + 1
                    else:
                        curr_count = visualidata.neu_service_at_other_banks[bank]
                        visualidata.neu_service_at_other_banks[bank] = curr_count + 1
                except Exception as e:
                    print(e)


        for bank in common_banks:
            visualidata.other_banks_total[bank] = visualidata.pos_service_at_other_banks[bank] + visualidata.neg_service_at_other_banks[bank] + visualidata.neu_service_at_other_banks[bank]
            print(bank, "-", visualidata.other_banks_total[bank])

    return visualidata


def create_json(request):
    common_banks = ['RBC', 'Scotiabank', 'CIBC', 'NBC', 'TD', 'BMO']
    # services = ["Credit", "Debit card", "Fee", "Rates", "Mortgage", "Online banking", "Customer Service", "Interest Rates", "Insurance", "Points", "Loan", "Interac", "Mobile banking", "Annual Fee", "Performance", "Security", "No Fee", "Rewards", "Yield", "Features", "Quick Access", "Mobile Deposit", "App Crash"]
    # services = ["Credit", "Debit card", "Fee", "Rates", "Mortgage", "Online banking"]
    services = ["something"]
    try:
        df = read_csv_from_gcs("text-mining-labeled-data", "final_labeled_reviews")
        json_objects = []

        for bank in common_banks:
            for service in services:
                visuali_data = analyze_service_sentiment(df, bank,  None)
                positive_text = " ".join(visuali_data.positive_reviews)
                negative_text = " ".join(visuali_data.negative_reviews)

                visuali_data.positive_word_list = generate_word_cloud_keyword_list(positive_text, sentiment='positive')
                visuali_data.negative_word_list = generate_word_cloud_keyword_list(negative_text, sentiment='negative')

                json_objects.append(visuali_data)
                print(bank, "-", service)

        save_models_to_json_file(json_objects, "json_database.json")

    except Exception as e:
        print(e)


############################################  utility functions  #######################################################


def save_models_to_json_file(models, filename):
    model_dicts = [model.to_dict() for model in models]

    # Serialize to JSON and write to file
    with open(filename, "w") as json_file:
        json.dump(model_dicts, json_file, indent=4)
    print(f"Models saved to {filename}")


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


def read_json_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    json_data = blob.download_as_text()
    data = json.loads(json_data)
    return data


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
    """
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
    if len(sentiment_words) == 0:
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color="white",
            colormap='Greens' if sentiment == 'positive' else 'Reds',
            max_words=1
        ).generate("No_Data_Found")
    else:
        # Initialize progress bar for word processing
        for word in tqdm(words, desc="Analyzing", unit="word"):
            if word not in stop_words and len(word) > 2:
                # Analyze sentiment of each word using RoBERTa
                # inputs = tokenizer(word, return_tensors="pt")
                # outputs = model(**inputs)
                # scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
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
    """
    return None


def generate_word_cloud_keyword_list(text, sentiment):
    """
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
    for word in words:
        if word not in stop_words and len(word) > 2:
            inputs = tokenizer(word, return_tensors="pt")
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            positive_score, neutral_score, negative_score = scores[2], scores[1], scores[0]

            # Filter words based on sentiment
            if sentiment == 'positive' and positive_score > 0.4:
                sentiment_words.append(word)
            elif sentiment == 'negative' and negative_score > 0.3:
                sentiment_words.append(word)

    return sentiment_words
    """
    return None

def fetch_data_all_bank_services(json_data):
    aggregated_data = {
        "positive_wordcloud": [],
        "negative_wordcloud": [],
    }
    for entry in json_data:
        bank = entry.get('bank_name')
        if bank not in aggregated_data:
            aggregated_data[bank] = {
                "total_reviews": 0,
                "avg_sentiment": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,

            }
            aggregated_data[bank]["total_reviews"] += entry["total_reviews"]
            aggregated_data[bank]["avg_sentiment"] = 0
            aggregated_data[bank]["positive_count"] += entry["pos_count"]
            aggregated_data[bank]["negative_count"] += entry["neg_count"]
            aggregated_data[bank]["neutral_count"] += entry["neu_count"]
            aggregated_data["positive_wordcloud"] += entry.get("positive_word_list", [])
            aggregated_data["negative_wordcloud"] += entry.get("negative_word_list", [])

    aggregated_data["positive_wordcloud"] = create_word_cloud_image(aggregated_data["positive_wordcloud"], 'positive')
    aggregated_data["negative_wordcloud"] = create_word_cloud_image(aggregated_data["negative_wordcloud"], 'negative')

    return aggregated_data


def overall_bank_sentiment_dashboard(request):

    # get all bank service names
    service_list = read_services_from_gcs("text-mining-labeled-data", "filtered_keywords.csv")
    # get all preprocessed banking details
    json_data = read_json_from_gcs("text-mining-labeled-data", "json_database.json")
    if json_data is not None:
        aggregated_data_json = fetch_data_all_bank_services(json_data)
        context = {
            "aggregated_data": aggregated_data_json,
            "service_list": service_list,
        }
        return render(request, 'BankSense/index_temp.html', context)

    else:
        # Extract from data from raw text0mining-labeled-data file and process it
        df = read_csv_from_gcs("text-mining-labeled-data", "final_labeled_reviews")
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
        positive_wordcloud = generate_word_cloud(top_positive_reviews_text, sentiment='positive')
        negative_wordcloud = generate_word_cloud(top_negative_reviews_text, sentiment='negative')
        # Convert the aggregated data to a dictionary format for the template
        aggregated_data_json = aggregated_data.to_dict(orient="records")
        service_list = read_services_from_gcs("text-mining-labeled-data", "filtered_keywords.csv")
        context = {
            "aggregated_data": aggregated_data_json,
            "service_list": service_list,
            "positive_wordcloud": positive_wordcloud,
            "negative_wordcloud": negative_wordcloud,
        }
        return render(request, 'BankSense/index_temp.html', context)
def summarize_reviews(reviews):
    """
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Iterate over each review in the list
    for i in range(len(reviews)):
        if len(reviews[i]) > 150:
            summary = summarizer(reviews[i], max_length=250, min_length=30, do_sample=False)
            reviews[i] = summary[0]['summary_text']

    return reviews
    """
    return None

def extract_sentiment_keywords(text, threshold=0.5):
    """
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
    """
    return None

def convert_dict_to_model(data):
    visuali_data = VisualiData()

    # Assign values directly
    visuali_data.bank_name = data.get("bank_name", "")
    visuali_data.total_reviews = data.get("total_reviews", 0)
    visuali_data.avg_rating = data.get("avg_rating", 0.0)
    visuali_data.searched_st_service = data.get("searched_st_service", "")
    visuali_data.searched_query = data.get("searched_query", "")

    visuali_data.positive_reviews = data.get("positive_reviews", [])
    visuali_data.negative_reviews = data.get("negative_reviews", [])
    visuali_data.pos_count = data.get("pos_count", 0)
    visuali_data.neg_count = data.get("neg_count", 0)
    visuali_data.neu_count = data.get("neu_count", 0)

    visuali_data.pos_service_at_other_banks = data.get("pos_service_at_other_banks", {})
    visuali_data.neg_service_at_other_banks = data.get("neg_service_at_other_banks", {})
    visuali_data.neu_service_at_other_banks = data.get("neu_service_at_other_banks", {})
    visuali_data.other_banks_total = data.get("other_banks_total", {})

    visuali_data.curr_bank_list = data.get("curr_bank_list", [])
    visuali_data.positive_word_list = data.get("positive_word_list", [])
    visuali_data.negative_word_list = data.get("negative_word_list", [])

    # Convert common_services list to ServiceModel instances
    common_services_data = data.get("common_services", [])

    for service in common_services_data:
        temp_model = ServiceModel()
        temp_model.name = service.get("name", "")
        temp_model.pos_count = service.get("pos_count", 0)
        temp_model.neg_count = service.get("neg_count", 0)
        temp_model.neu_count = service.get("neu_count", 0)

        visuali_data.common_services.append(temp_model)

    return visuali_data


def create_word_cloud_image(sentiment_words, sentiment):
    offensive_words = [
        "f***", "f***er", "f***ing", "fuck", "fucker", "fucking",
        "s***", "s***ty", "shit", "shitty",
        "b****", "b***ard", "bitch", "bastard",
        "a**", "a***hole", "ass", "asshole",
        "d***", "d***head", "damn", "dick", "dickhead",
        "c***", "cunt",
        "p****", "pussy",
        "w****", "whore",
        "s***", "slut",
        "t***", "tits",
        "v****", "vagina",
        "j****", "jerk", "b******", "blowjob",
        "hell",
        "lmao", "wtf",
        "suck", "s***", "s*ck", "s**k", "nigga", "n****", "fuking",
        "coronavirus", "corona", "unemployment", "supbar", "standard"
    ]

    sentiment_words = [
        word for word in sentiment_words
        if word not in offensive_words
    ]
    if len(sentiment_words) == 0:
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color="white",
            colormap='Greens' if sentiment == 'positive' else 'Reds',
            max_words=50
        ).generate("No Data Found")
    else:
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color="white",
            colormap='Greens' if sentiment == 'positive' else 'Reds',
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


def fetch_data_by_bank_and_service(json_data, bank_name, service_name):
    if service_name is not None:
        service_name = ' '.join(word.replace('-', ' ') for word in service_name.split())

        for entry in json_data:
            # Check if the dictionary contains the matching bank_name and service_name
            if entry.get("bank_name") == bank_name and entry.get("searched_st_service") == service_name:
                return entry

    else:
        for entry in json_data:
            # Check if the dictionary contains the matching bank_name
            if entry.get("bank_name") == bank_name and entry.get("searched_st_service") == "":
                return entry

    return None
