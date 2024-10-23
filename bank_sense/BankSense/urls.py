from django.urls import path
from BankSense import views


app_name = 'BankSense'

urlpatterns = [
    # path('', views.review_list, name='index'),
    path('', views.dashboard_view, name='dashboard'),
    path('store_data_in_db', views.store_data_in_db, name='store_data_in_db'),
]