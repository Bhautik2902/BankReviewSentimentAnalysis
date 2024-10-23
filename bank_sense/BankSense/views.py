import os
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .models import Review, VisuliData
from datetime import datetime


# View to list all reviews
def dashboard_view(request):
    file_path = os.path.join(settings.BASE_DIR, 'BankSense', 'data', 'reddit_google_merged_data.xlsx')

    service_name = 'Credit' #request.GET.get('service', None)
    bank_name = 'RBC' # request.GET.get('bank', None)
    search_query = '', # request.GET.get('query', None)

    try:
        df = pd.read_excel(file_path)
        analyze_service_sentiment(df, bank_name, service_name, search_query)

    except Exception as e:
        print(str(e))

    return render(request, 'BankSense/index.html')


def analyze_service_sentiment(df, bank_name, service_name, search_query=None):
    # Initialize variables
    sentiment_counter = 0
    positive_reviews = []
    negative_reviews = []
    keywords_to_avoid = ['app', 'interface', 'ui', 'layout', 'design', 'update', 'fingertips', 'bug', 'fingerprint', 'version']

    # Filter reviews by the given bank name and check for the service name in the review
    bank_reviews = df[df['AppName'].str.split('_').str[0] == bank_name]

    for _, row in bank_reviews.iterrows():
        review = row['Review']
        review = str(review).lower()

        if service_name.lower() in review:
            sentiment = row['Review sentiment']

            # Increment or decrement sentiment_counter based on the sentiment
            if sentiment == 'positive' and all(keyword not in review for keyword in keywords_to_avoid):
                sentiment_counter += 1
                if len(positive_reviews) < 5:  # Record max 5 positive reviews
                    positive_reviews.append(review)
            elif sentiment == 'negative' and all(keyword not in review for keyword in keywords_to_avoid):
                sentiment_counter -= 1
                if len(negative_reviews) < 5:
                    negative_reviews.append(review)

    return sentiment_counter, positive_reviews, negative_reviews


def store_data_in_db(request):
    response = HttpResponse()
    response.write("Already stored")
    return response
#     file_path = os.path.join(settings.BASE_DIR, 'BankSense', 'data', 'reddit_google_merged_data.xlsx')
#     response.write(file_path)
#
#     try:
#         df = pd.read_excel(file_path)
#     except Exception as e:
#         response.write(str(e))
#
#     for index, row in df.iterrows():
#         # Convert the date field to a Python datetime object if necessary
#         date_str = row['date']
#         if pd.isnull(date_str):
#             date_obj = None
#         else:
#             try:
#                 date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M')  # Adjust date format if needed
#             except ValueError:
#                 date_obj = None  # Handle any unexpected date formats
#
#         # Create and save the review object
#         review = Review(
#             index=row['Index'],
#             source=row['source'],
#             bank=row['bank'],
#             title=row['title'],
#             review_text=str(row['review_text']),  # Convert to string
#             rating=row['rating'] if not pd.isnull(row['rating']) else None,
#             date=date_obj,
#             url=row['url'] if not pd.isnull(row['url']) else None,
#             source_type=row['source_type'] if not pd.isnull(row['source_type']) else None,
#             review_sentiment=row['review_sentiment'],
#             sentiment_score=row['sentiment_score'],
#         )
#         review.save()
#
#     response.write("review stored successfully")
#     return response
