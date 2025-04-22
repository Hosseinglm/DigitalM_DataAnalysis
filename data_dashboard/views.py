from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import data_analysis
import pandas as pd
import json
import plotly.graph_objects as go

# Create your views here.
def index(request):
    """Home page view with integrated metrics from all analyses"""
    # Load all data
    datasets = data_analysis.load_data()
    
    if not datasets:
        # Handle the case where data loading failed
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Run analyses to get overall metrics
    ga_results = data_analysis.analyze_google_analytics(datasets['ga_data'])
    customer_results = data_analysis.analyze_customer_data(datasets['customer_data'])
    ad_results = data_analysis.analyze_ad_performance(
        datasets['fb_ads'], datasets['ig_ads'], datasets['google_ads']
    )
    
    # Calculate blended metrics
    overall_kpis = data_analysis.calculate_overall_metrics(
        ga_results, ad_results, customer_results
    )
    
    # Prepare key figures for the overview
    top_kpis = {
        'Total Revenue': overall_kpis.get('Total Revenue', '$0'),
        'Blended ROAS': overall_kpis.get('Blended ROAS', '0'),
        'Total Sessions': overall_kpis.get('Total Sessions', '0'),
        'Total Customers': overall_kpis.get('Total Customers', '0'),
        'Total Ad Spend': overall_kpis.get('Total Ad Spend', '$0'),
        'AOV': overall_kpis.get('AOV', '$0'),
        'CAC': overall_kpis.get('CAC', '$0'),
        'LTV Proxy': overall_kpis.get('LTV Proxy', '$0')
    }
    
    # Convert key trend charts to HTML for the overview page
    sessions_transactions_chart_html = ga_results['fig_daily_trends'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    ad_performance_chart_html = ad_results['fig_daily_ads'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    context = {
        'top_kpis': top_kpis,
        'sessions_transactions_chart': sessions_transactions_chart_html,
        'ad_performance_chart': ad_performance_chart_html,
    }
    
    return render(request, 'data_dashboard/index.html', context)

def google_analytics(request):
    """Google Analytics dashboard view using the refactored analysis"""
    # Load data
    datasets = data_analysis.load_data()
    
    if not datasets:
        # Handle the case where data loading failed
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Run the full GA analysis using the refactored function
    ga_results = data_analysis.analyze_google_analytics(datasets['ga_data'])
    
    # Convert Plotly figures to HTML to embed in the template
    sessions_by_source_chart_html = ga_results['fig_sessions_source'].to_html(
        full_html=False, include_plotlyjs=False
    )
    sessions_transactions_chart_html = ga_results['fig_daily_trends'].to_html(
        full_html=False, include_plotlyjs=False
    )
    sessions_by_device_chart_html = ga_results['fig_device_sessions'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    # Convert DataFrames to lists of dicts for the template
    sessions_by_source_data = ga_results['sessions_by_source_df'].to_dict('records')
    device_metrics_data = ga_results['device_metrics_df'].to_dict('records')
    
    context = {
        'ga_kpis': ga_results['kpis'],
        'sessions_by_source_chart': sessions_by_source_chart_html,
        'sessions_transactions_chart': sessions_transactions_chart_html,
        'sessions_by_device_chart': sessions_by_device_chart_html,
        'sessions_by_source_data': sessions_by_source_data,
        'device_metrics_data': device_metrics_data
    }
    
    return render(request, 'data_dashboard/google_analytics.html', context)

def customer_analysis(request):
    """Customer analysis dashboard view using the refactored analysis"""
    # Load data
    datasets = data_analysis.load_data()
    
    if not datasets:
        # Handle the case where data loading failed
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Run the full customer analysis using the refactored function
    customer_results = data_analysis.analyze_customer_data(datasets['customer_data'])
    
    # Convert Plotly figures to HTML to embed in the template
    customer_segments_chart_html = customer_results['fig_customer_segments'].to_html(
        full_html=False, include_plotlyjs=False
    )
    age_distribution_chart_html = customer_results['fig_age_distribution'].to_html(
        full_html=False, include_plotlyjs=False
    )
    loyalty_tier_chart_html = customer_results['fig_loyalty_tier'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    # Convert DataFrames to lists of dicts for the template
    segment_counts_data = customer_results['segment_counts_df'].to_dict('records')
    loyalty_analysis_data = customer_results['loyalty_analysis_df'].to_dict('records')
    
    context = {
        'customer_kpis': customer_results['kpis'],
        'gender_distribution': customer_results['gender_distribution'],
        'customer_segments_chart': customer_segments_chart_html,
        'age_distribution_chart': age_distribution_chart_html,
        'loyalty_tier_chart': loyalty_tier_chart_html,
        'segment_counts_data': segment_counts_data,
        'loyalty_analysis_data': loyalty_analysis_data
    }
    
    return render(request, 'data_dashboard/customer_analysis.html', context)

def ad_performance(request):
    """Ad performance dashboard view using the refactored analysis"""
    # Load data
    datasets = data_analysis.load_data()
    
    if not datasets:
        # Handle the case where data loading failed
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Run the full ad performance analysis using the refactored function
    ad_results = data_analysis.analyze_ad_performance(
        datasets['fb_ads'], datasets['ig_ads'], datasets['google_ads']
    )
    
    # Convert Plotly figures to HTML to embed in the template
    platform_roas_chart_html = ad_results['fig_platform_roas'].to_html(
        full_html=False, include_plotlyjs=False
    )
    platform_conv_chart_html = ad_results['fig_platform_conv'].to_html(
        full_html=False, include_plotlyjs=False
    )
    spend_value_chart_html = ad_results['fig_spend_value'].to_html(
        full_html=False, include_plotlyjs=False
    )
    daily_ads_chart_html = ad_results['fig_daily_ads'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    # Convert DataFrames to lists of dicts for the template
    platform_summary_data = ad_results['platform_summary_df'].to_dict('records')
    
    context = {
        'platform_summary_data': platform_summary_data,
        'platform_roas_chart': platform_roas_chart_html,
        'platform_conv_chart': platform_conv_chart_html,
        'spend_value_chart': spend_value_chart_html,
        'daily_ads_chart': daily_ads_chart_html
    }
    
    return render(request, 'data_dashboard/ad_performance.html', context)

# New views for the enhanced dashboard features

def attribution(request):
    """Attribution modeling dashboard view"""
    # Load data
    datasets = data_analysis.load_data()
    
    if not datasets:
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Run attribution analysis (placeholder for now)
    attribution_results = data_analysis.perform_attribution_analysis(datasets['touchpoints'])
    
    # Convert to HTML/JSON
    sankey_chart_html = attribution_results['sankey_fig'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    context = {
        'sankey_chart': sankey_chart_html,
        'attribution_data': attribution_results['attribution_results'].to_dict('records') if not attribution_results['attribution_results'].empty else [],
        'is_placeholder': True  # Flag to show this feature is under development
    }
    
    return render(request, 'data_dashboard/attribution.html', context)

def cohort_analysis(request):
    """Cohort analysis dashboard view"""
    # Load data
    datasets = data_analysis.load_data()
    
    if not datasets:
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Run cohort analysis (placeholder for now)
    cohort_results = data_analysis.perform_cohort_analysis(datasets['customer_data'])
    
    # Convert to HTML/JSON
    cohort_heatmap_html = cohort_results['cohort_heatmap_fig'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    context = {
        'cohort_heatmap': cohort_heatmap_html,
        'cohort_data': cohort_results['cohort_data'].to_dict('records') if not cohort_results['cohort_data'].empty else [],
        'is_placeholder': True  # Flag to show this feature is under development
    }
    
    return render(request, 'data_dashboard/cohort_analysis.html', context)

def forecast(request):
    """Forecasting dashboard view"""
    # Load data
    datasets = data_analysis.load_data()
    
    if not datasets:
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Get the time series data from GA
    ga_results = data_analysis.analyze_google_analytics(datasets['ga_data'])
    
    # Get parameters from request, with defaults
    forecast_horizon = request.GET.get('horizon', '30')
    forecast_metric = request.GET.get('metric', 'revenue')
    
    # Convert horizon to integer
    try:
        horizon_days = int(forecast_horizon)
    except ValueError:
        horizon_days = 30
    
    # Run forecast with requested parameters
    forecast_results = data_analysis.generate_forecast(
        ga_results['daily_metrics_df'], 
        forecast_horizon=horizon_days,
        forecast_metric=forecast_metric
    )
    
    # Convert to HTML/JSON
    forecast_chart_html = forecast_results['forecast_fig'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    context = {
        'forecast_chart': forecast_chart_html,
        'forecast_data': forecast_results['forecast_df'].to_dict('records') if not forecast_results['forecast_df'].empty else [],
        'forecast_summary': forecast_results['forecast_summary'],
        'selected_horizon': horizon_days,
        'selected_metric': forecast_metric,
        'is_placeholder': True  # Flag to show this feature is under development
    }
    
    return render(request, 'data_dashboard/forecast.html', context)

def budget_recommendations(request):
    """Budget recommendations dashboard view"""
    # Load data
    datasets = data_analysis.load_data()
    
    if not datasets:
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Get ad performance data
    ad_results = data_analysis.analyze_ad_performance(
        datasets['fb_ads'], datasets['ig_ads'], datasets['google_ads']
    )
    
    # Get parameters from request, with defaults
    total_budget = request.GET.get('budget', '50000')
    primary_goal = request.GET.get('goal', 'revenue')
    optimization_method = request.GET.get('method', 'roas')
    
    # Convert budget to float
    try:
        budget_amount = float(total_budget)
    except ValueError:
        budget_amount = 50000
    
    # Run budget recommendations with requested parameters
    budget_results = data_analysis.generate_budget_recommendations(
        ad_results['platform_summary_df'],
        total_budget=budget_amount,
        primary_goal=primary_goal,
        optimization_method=optimization_method
    )
    
    # Convert charts to HTML
    budget_chart_html = budget_results['budget_chart'].to_html(
        full_html=False, include_plotlyjs=False
    )
    
    change_chart_html = budget_results.get('change_chart', go.Figure()).to_html(
        full_html=False, include_plotlyjs=False
    ) if 'change_chart' in budget_results else ''
    
    metrics_chart_html = budget_results.get('metrics_chart', go.Figure()).to_html(
        full_html=False, include_plotlyjs=False
    ) if 'metrics_chart' in budget_results else ''
    
    context = {
        'budget_chart': budget_chart_html,
        'change_chart': change_chart_html,
        'metrics_chart': metrics_chart_html,
        'allocation_data': budget_results['allocation_data'].to_dict('records') if not budget_results['allocation_data'].empty else [],
        'recommendations': budget_results['recommendations'],
        'insights': budget_results.get('insights', []),
        'expected_outcomes': budget_results.get('expected_outcomes', {}),
        'selected_budget': budget_amount,
        'selected_goal': primary_goal,
        'selected_method': optimization_method,
        'is_placeholder': True  # Flag to show this feature is under development
    }
    
    return render(request, 'data_dashboard/budget_recommendations.html', context)

def dashboard(request):
    """Integrated dashboard view with all metrics and features"""
    # Load all data
    datasets = data_analysis.load_data()
    
    if not datasets:
        error_context = {
            'error_message': 'Failed to load data. Please check that all required CSV files exist.'
        }
        return render(request, 'data_dashboard/error.html', error_context)
    
    # Run all analyses
    ga_results = data_analysis.analyze_google_analytics(datasets['ga_data'])
    customer_results = data_analysis.analyze_customer_data(datasets['customer_data'])
    ad_results = data_analysis.analyze_ad_performance(
        datasets['fb_ads'], datasets['ig_ads'], datasets['google_ads']
    )
    
    # Calculate blended metrics
    overall_kpis = data_analysis.calculate_overall_metrics(
        ga_results, ad_results, customer_results
    )
    
    # Run placeholder analyses for future features
    attribution_results = data_analysis.perform_attribution_analysis(datasets['touchpoints'])
    cohort_results = data_analysis.perform_cohort_analysis(datasets['customer_data'])
    forecast_results = data_analysis.generate_forecast(ga_results['daily_metrics_df'])
    budget_results = data_analysis.generate_budget_recommendations(ad_results['platform_summary_df'])
    
    # Convert all charts to HTML
    chart_html = {
        # GA charts
        'sessions_source': ga_results['fig_sessions_source'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'daily_trends': ga_results['fig_daily_trends'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'device_sessions': ga_results['fig_device_sessions'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        
        # Customer charts
        'customer_segments': customer_results['fig_customer_segments'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'age_distribution': customer_results['fig_age_distribution'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'loyalty_tier': customer_results['fig_loyalty_tier'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        
        # Ad performance charts
        'platform_roas': ad_results['fig_platform_roas'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'platform_conv': ad_results['fig_platform_conv'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'spend_value': ad_results['fig_spend_value'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'daily_ads': ad_results['fig_daily_ads'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        
        # Placeholder charts for future features
        'sankey': attribution_results['sankey_fig'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'cohort_heatmap': cohort_results['cohort_heatmap_fig'].to_html(
            full_html=False, include_plotlyjs=False
        ),
        'forecast': forecast_results['forecast_fig'].to_html(
            full_html=False, include_plotlyjs=False
        )
    }
    
    context = {
        'overall_kpis': overall_kpis,
        'charts': chart_html,
        'ga_results': {
            'kpis': ga_results['kpis'],
            'sessions_by_source_data': ga_results['sessions_by_source_df'].to_dict('records'),
            'device_metrics_data': ga_results['device_metrics_df'].to_dict('records')
        },
        'customer_results': {
            'kpis': customer_results['kpis'],
            'segment_counts_data': customer_results['segment_counts_df'].to_dict('records'),
            'loyalty_analysis_data': customer_results['loyalty_analysis_df'].to_dict('records')
        },
        'ad_results': {
            'platform_summary_data': ad_results['platform_summary_df'].to_dict('records')
        },
        'budget_recommendations': budget_results['recommendations']
    }
    
    return render(request, 'data_dashboard/dashboard.html', context)