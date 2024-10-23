import os
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .models import Review
from datetime import datetime


# View to list all reviews
def dashboard_view(request):
    return render(request, 'BankSense/index.html')


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
