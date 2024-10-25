import os
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .models import Review, VisualiData, ServiceModel
from datetime import datetime


# View to list all reviews
def dashboard_view(request):
    file_path = os.path.join(settings.BASE_DIR, 'BankSense', 'data', 'reddit_google_merged_data.xlsx')

    service_name = request.GET.get('service', None)
    bank_name = request.GET.get('bank', None)
    search_query = request.GET.get('query', None)

    try:
        df = pd.read_excel(file_path)
        visuali_data = analyze_service_sentiment(df, bank_name, service_name, search_query)

    except Exception as e:
        print(str(e))

    return render(request, 'BankSense/index.html', {'visuali_data': visuali_data})


def analyze_service_sentiment(df, bank_name, service_name, search_query=None):

    keywords_to_avoid = ['app', 'interface', 'ui', 'layout', 'design', 'update', 'fingertips', 'bug', 'fingerprint', 'version']
    common_st_services = ['Credit', 'Security', 'Online banking', 'Mortgage', 'fee']

    visulidata = VisualiData()
    visulidata.bank_name = bank_name

    # assigning top 5 services to visulidata
    for service in common_st_services:
        servicemodel = ServiceModel()
        servicemodel.name = service
        visulidata.common_services.append(servicemodel)

    # Filter reviews by the given bank name and check for the service name in the review
    bank_reviews = df[df['bank'] == bank_name]

    visulidata.bank_name = bank_name

    for service in visulidata.common_services:
        print(service.name)
        for _, row in bank_reviews.iterrows():
            review = row['review_text']
            review = str(review).lower()

            if service.name.lower() in review:
                sentiment = row['review_sentiment']

                # Increment or decrement sentiment_counter based on the sentiment
                if sentiment == 'positive' and all(keyword not in review for keyword in keywords_to_avoid):
                    service.pos_count += 1
                    if len(visulidata.positive_reviews) < 5:
                        visulidata.positive_reviews.append(review)
                elif sentiment == 'negative' and all(keyword not in review for keyword in keywords_to_avoid):
                    service.neg_count += 1
                    if len(visulidata.negative_reviews) < 5:
                        visulidata.negative_reviews.append(review)
                elif sentiment == 'neutral' and all(keyword not in review for keyword in keywords_to_avoid):
                    service.neu_count += 1

    return visulidata


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
