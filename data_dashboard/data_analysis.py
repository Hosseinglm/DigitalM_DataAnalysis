import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from datetime import datetime, timedelta
import warnings
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Load datasets
def load_data():
    """Load all datasets from the data directory"""
    
    # Google Analytics data
    ga_data = pd.read_csv('data/google_analytics_data.csv')
    ga_data['date'] = pd.to_datetime(ga_data['date'])
    
    # Customer data
    customer_data = pd.read_csv('data/customer_data.csv')
    customer_data['signup_date'] = pd.to_datetime(customer_data['signup_date'])
    customer_data['last_purchase_date'] = pd.to_datetime(customer_data['last_purchase_date'])
    
    # Social media ads data
    fb_ads = pd.read_csv('data/facebook_ads_data.csv')
    fb_ads['date'] = pd.to_datetime(fb_ads['date'])
    
    ig_ads = pd.read_csv('data/instagram_ads_data.csv')
    ig_ads['date'] = pd.to_datetime(ig_ads['date'])
    
    # Google Ads data
    google_ads = pd.read_csv('data/google_ads_data.csv')
    google_ads['date'] = pd.to_datetime(google_ads['date'])
    
    # Customer touchpoints
    touchpoints = pd.read_csv('data/customer_touchpoints.csv')
    touchpoints['date'] = pd.to_datetime(touchpoints['date'])
    
    # Google Search Console data
    search_console = pd.read_csv('data/google_search_console_data.csv')
    search_console['date'] = pd.to_datetime(search_console['date'])
    
    return {
        'ga_data': ga_data,
        'customer_data': customer_data,
        'fb_ads': fb_ads,
        'ig_ads': ig_ads,
        'google_ads': google_ads,
        'touchpoints': touchpoints,
        'search_console': search_console
    }

def get_google_analytics_stats(ga_data):
    """Get basic Google Analytics statistics"""
    stats = {
        'total_sessions': ga_data['sessions'].sum(),
        'total_new_users': ga_data['new_users'].sum(),
        'total_transactions': ga_data['transactions'].sum(),
        'total_revenue': ga_data['revenue'].sum(),
        'avg_conversion_rate': ga_data['conversion_rate'].mean() * 100,  # as percentage
    }
    return stats

def get_sessions_by_source_chart(ga_data):
    """Create a bar chart of sessions by source"""
    sessions_by_source = ga_data.groupby('source')['sessions'].sum().reset_index()
    sessions_by_source = sessions_by_source.sort_values('sessions', ascending=False)
    
    fig = px.bar(
        sessions_by_source, 
        x='source', 
        y='sessions',
        title='Sessions by Source',
        labels={'source': 'Source', 'sessions': 'Total Sessions'},
        color='sessions',
        color_continuous_scale='Viridis'
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_sessions_transactions_time_chart(ga_data):
    """Create a line chart of sessions and transactions over time"""
    daily_metrics = ga_data.groupby('date').agg({
        'sessions': 'sum',
        'transactions': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['date'], 
            y=daily_metrics['sessions'],
            name='Sessions',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['date'], 
            y=daily_metrics['transactions'],
            name='Transactions',
            line=dict(color='green')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Sessions and Transactions Over Time',
        xaxis_title='Date',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
    )
    
    fig.update_yaxes(title_text="Sessions", secondary_y=False)
    fig.update_yaxes(title_text="Transactions", secondary_y=True)
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_sessions_by_device_chart(ga_data):
    """Create a pie chart of sessions by device"""
    device_metrics = ga_data.groupby('device_category').agg({
        'sessions': 'sum',
        'transactions': 'sum',
        'conversion_rate': 'mean'
    }).reset_index()
    
    fig = px.pie(
        device_metrics, 
        values='sessions', 
        names='device_category',
        title='Sessions by Device Category',
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_customer_demographics(customer_data):
    """Get customer demographic information"""
    demographics = {
        'total_customers': len(customer_data),
        'gender_distribution': customer_data['gender'].value_counts(normalize=True).to_dict(),
        'avg_age': customer_data['age'].mean(),
        'avg_total_spend': customer_data['total_spend'].mean(),
    }
    return demographics

def get_customer_segments_chart(customer_data):
    """Create a bar chart of customer segments"""
    segment_counts = customer_data['customer_segment'].value_counts()
    
    fig = px.bar(
        x=segment_counts.index, 
        y=segment_counts.values,
        title='Customer Segments',
        labels={'x': 'Segment', 'y': 'Number of Customers'},
        color=segment_counts.values,
        color_continuous_scale='Viridis'
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_age_distribution_chart(customer_data):
    """Create a histogram of customer ages"""
    fig = px.histogram(
        customer_data, 
        x='age',
        nbins=20,
        title='Customer Age Distribution',
        labels={'age': 'Age', 'count': 'Number of Customers'},
        color_discrete_sequence=['purple']
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_loyalty_tier_chart(customer_data):
    """Create a grouped bar chart of loyalty tiers"""
    loyalty_spend = customer_data.groupby('loyalty_tier').agg({
        'total_spend': 'mean',
        'total_orders': 'mean'
    }).reset_index()
    
    fig = px.bar(
        loyalty_spend, 
        x='loyalty_tier', 
        y=['total_spend', 'total_orders'],
        barmode='group',
        title='Average Spend and Orders by Loyalty Tier',
        labels={
            'loyalty_tier': 'Loyalty Tier', 
            'value': 'Value',
            'variable': 'Metric'
        }
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_platform_performance(fb_ads, ig_ads, google_ads):
    """Get ad performance metrics across platforms"""
    # Combine ad data
    fb_ads_summary = fb_ads.groupby('date').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'conversions': 'sum',
        'conversion_value': 'sum'
    }).reset_index()
    fb_ads_summary['platform'] = 'Facebook'
    
    ig_ads_summary = ig_ads.groupby('date').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'conversions': 'sum',
        'conversion_value': 'sum'
    }).reset_index()
    ig_ads_summary['platform'] = 'Instagram'
    
    google_ads_summary = google_ads.groupby('date').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'cost': 'sum',
        'conversions': 'sum',
        'conversion_value': 'sum'
    }).reset_index()
    google_ads_summary = google_ads_summary.rename(columns={'cost': 'spend'})
    google_ads_summary['platform'] = 'Google Ads'
    
    # Combine all platforms
    all_ads = pd.concat([
        fb_ads_summary, 
        ig_ads_summary, 
        google_ads_summary
    ], ignore_index=True)
    
    # Calculate metrics
    all_ads['ctr'] = all_ads['clicks'] / all_ads['impressions']
    all_ads['cpc'] = all_ads['spend'] / all_ads['clicks']
    all_ads['roas'] = all_ads['conversion_value'] / all_ads['spend']
    
    # Platform performance comparison
    platform_performance = all_ads.groupby('platform').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'conversions': 'sum',
        'conversion_value': 'sum',
        'ctr': 'mean',
        'cpc': 'mean',
        'roas': 'mean'
    }).reset_index()
    
    return platform_performance

def get_platform_conversions_chart(fb_ads, ig_ads, google_ads):
    """Create a bar chart comparing platform conversions"""
    platform_performance = get_platform_performance(fb_ads, ig_ads, google_ads)
    
    fig = px.bar(
        platform_performance, 
        x='platform', 
        y='conversions',
        color='platform',
        title='Conversions by Platform',
        labels={'platform': 'Platform', 'conversions': 'Total Conversions'}
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_platform_roas_chart(fb_ads, ig_ads, google_ads):
    """Create a bar chart comparing platform ROAS"""
    platform_performance = get_platform_performance(fb_ads, ig_ads, google_ads)
    
    fig = px.bar(
        platform_performance, 
        x='platform', 
        y='roas',
        color='platform',
        title='Return on Ad Spend (ROAS) by Platform',
        labels={'platform': 'Platform', 'roas': 'ROAS'}
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_daily_conversions_chart(fb_ads, ig_ads, google_ads):
    """Create a line chart of daily conversions by platform"""
    # Combine ad data
    fb_ads_summary = fb_ads.groupby('date').agg({
        'conversions': 'sum',
    }).reset_index()
    fb_ads_summary['platform'] = 'Facebook'
    
    ig_ads_summary = ig_ads.groupby('date').agg({
        'conversions': 'sum',
    }).reset_index()
    ig_ads_summary['platform'] = 'Instagram'
    
    google_ads_summary = google_ads.groupby('date').agg({
        'conversions': 'sum',
    }).reset_index()
    google_ads_summary['platform'] = 'Google Ads'
    
    # Combine all platforms
    all_ads = pd.concat([
        fb_ads_summary, 
        ig_ads_summary, 
        google_ads_summary
    ], ignore_index=True)
    
    fig = px.line(
        all_ads, 
        x='date', 
        y='conversions',
        color='platform',
        title='Daily Conversions by Platform',
        labels={'date': 'Date', 'conversions': 'Conversions', 'platform': 'Platform'}
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
