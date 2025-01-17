from django.urls import path
from BankSense import views


app_name = 'BankSense'

urlpatterns = [
    # path('', views.review_list, name='index'),
    path('', views.dashboard_view, name='dashboard'),

    path('overview', views.overall_bank_sentiment_dashboard, name='overview_dashboard'),
    path('create_json', views.create_json, name='create_json'),

]