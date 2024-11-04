import os
from datetime import datetime

import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .models import Review, VisualiData, ServiceModel
from django.http import JsonResponse
import io
from google.cloud import storage
from transformers import pipeline


# View to list all reviews
def dashboard_view(request):

    service_name = request.GET.get('service', None)
    bank_name = request.GET.get('bank', 'CIBC')

    try:
        df = read_csv_from_gcs("text-mining-labeled-data", "labeled_reviews1")
        service_list = read_services_from_gcs("text-mining-labeled-data", "filtered_keywords.csv")

        visuali_data = analyze_service_sentiment(df, bank_name, service_name)
        service_list.remove('Keyword')
        return render(request, 'BankSense/index.html', {'visuali_data': visuali_data, 'service_list': service_list})

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
            visuali_data.positive_reviews = df[(df["bank"] == row["bank"]) & (df["predicted_sentiment"] == "positive")]["review_text"].tolist()
            visuali_data.negative_reviews = df[(df["bank"] == row["bank"]) & (df["predicted_sentiment"] == "negative")]["review_text"].tolist()

            # Assuming `common_services` would be populated by additional logic or ServiceModel instances
            # For now, it's left as an empty list in the VisualiData instance
    print("this is visuali_data"+str(visuali_data))
    return visuali_data
def analyze_service_sentiment(df, bank_name, service_name=None):
    keywords_to_avoid = ['app', 'interface', 'ui', 'layout', 'design', 'update', 'fingertips', 'bug', 'fingerprint',
                         'version']
    common_st_services = ['Credit', 'Security', 'Online banking', 'Mortgage', 'Fee']


    if service_name is not None:
        common_st_services.insert(0, service_name.replace('-', ' '))  # add selected service at front
        common_st_services.pop() # remove last one.

    visualidata = VisualiData()
    visualidata.bank_name = bank_name

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
                if sentiment == 'positive':  #and all(keyword not in review for keyword in keywords_to_avoid):
                    service.pos_count += 1

                    if len(visualidata.positive_reviews) < 5:
                        visualidata.positive_reviews.append(review)
                elif sentiment == 'negative':  #and all(keyword not in review for keyword in keywords_to_avoid):
                    service.neg_count += 1
                    if len(visualidata.negative_reviews) < 5:
                        visualidata.negative_reviews.append(review)
                elif sentiment == 'neutral':  #and all(keyword not in review for keyword in keywords_to_avoid):
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

    return visualidata


############################################  utility functions  #######################################################


def store_data_in_db(request):
    response = HttpResponse()

    file_path = os.path.join(settings.BASE_DIR, 'BankSense', 'data', 'reddit_google_merged_data.xlsx')
    response.write(file_path)

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        response.write(str(e))

    for index, row in df.iterrows():
        # Convert the date field to a Python datetime object if necessary
        date_str = row['date']
        if pd.isnull(date_str):
            date_obj = None
        else:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M')  # Adjust date format if needed
            except ValueError:
                date_obj = None  # Handle any unexpected date formats

        # Create and save the review object
        review = Review(
            index=row['Index'],
            source=row['source'],
            bank=row['bank'],
            title=row['title'],
            review_text=str(row['review_text']),  # Convert to string
            rating=row['rating'] if not pd.isnull(row['rating']) else None,
            date=date_obj,
            url=row['url'] if not pd.isnull(row['url']) else None,
            source_type=row['source_type'] if not pd.isnull(row['source_type']) else None,
            review_sentiment=row['review_sentiment'],
            sentiment_score=row['sentiment_score'],
        )
        review.save()

    response.write("review stored successfully")
    return response


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


def test_gcs_access(request):

    try:
        # Verify the environment variable is correctly set

        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            return JsonResponse({'status': 'error', 'message': 'GOOGLE_APPLICATION_CREDENTIALS not set'}, status=400)

        # Initialize Google Cloud Storage client
        storage_client = storage.Client()

        # Specify your bucket name
        bucket_name = 'your-bucket-name'  # Replace with your actual bucket name
        bucket = storage_client.get_bucket(bucket_name)

        # List all files (blobs) in the bucket and their paths
        file_paths = [blob.name for blob in bucket.list_blobs()]

        # Construct full GCS paths for each file
        gcs_file_paths = [f"gs://{bucket_name}/{file_path}" for file_path in file_paths]

        # Return the file paths
        return JsonResponse({'status': 'success', 'file_paths': gcs_file_paths})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def overall_bank_sentiment_dashboard(request):
    df = read_csv_from_gcs("text-mining-labeled-data", "labeled_reviews1")
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
    }
    return render(request, 'BankSense/index_temp.html', context)
