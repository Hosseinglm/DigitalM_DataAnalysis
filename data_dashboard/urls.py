from django.urls import path
from . import views

app_name = 'data_dashboard'

urlpatterns = [
    path('', views.index, name='index'),
    path('google-analytics/', views.google_analytics, name='google_analytics'),
    path('customer-analysis/', views.customer_analysis, name='customer_analysis'),
    path('ad-performance/', views.ad_performance, name='ad_performance'),
    
    # New URLs for enhanced dashboard features
    path('dashboard/', views.dashboard, name='dashboard'),
    path('attribution/', views.attribution, name='attribution'),
    path('cohort-analysis/', views.cohort_analysis, name='cohort_analysis'),
    path('forecast/', views.forecast, name='forecast'),
    path('budget-recommendations/', views.budget_recommendations, name='budget_recommendations'),
]
