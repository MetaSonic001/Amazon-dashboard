import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Load and prepare data
def load_data():
    # Load your generated datasets
    products = pd.read_csv('amazon_products_data.csv')
    users = pd.read_csv('amazon_users_data.csv')
    events = pd.read_csv('amazon_events_data.csv')
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    return products, users, events

products_df, users_df, events_df = load_data()

# Create layout with multiple tabs
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Amazon Analytics Dashboard", className="text-primary text-center mb-4"),
            html.P("Comprehensive analysis of user behavior and recommendations", className="text-center mb-4")
        ])
    ]),
    
    dbc.Tabs([
        # Tab 1: Overview Dashboard
        dbc.Tab(label="Overview", children=[
            dbc.Row([
                # KPI Cards
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Users", className="card-title"),
                            html.H2(id="total-users", className="text-primary")
                        ])
                    ], className="mb-4")
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Prime Members", className="card-title"),
                            html.H2(id="prime-members", className="text-success")
                        ])
                    ], className="mb-4")
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Sales", className="card-title"),
                            html.H2(id="total-sales", className="text-info")
                        ])
                    ], className="mb-4")
                ], width=4),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sales Trend", className="card-title"),
                            dcc.Graph(id="sales-trend")
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Category Distribution", className="card-title"),
                            dcc.Graph(id="category-pie")
                        ])
                    ])
                ], width=4),
            ], className="mb-4"),
        ]),
        
        # Tab 2: Prime Analysis
        dbc.Tab(label="Prime Analysis", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Prime vs Non-Prime Purchases", className="card-title"),
                            dcc.Graph(id="prime-comparison")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Prime Benefits Usage", className="card-title"),
                            dcc.Graph(id="prime-benefits")
                        ])
                    ])
                ], width=6),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Prime Member Retention", className="card-title"),
                            dcc.Graph(id="prime-retention")
                        ])
                    ])
                ])
            ]),
        ]),
        
        # Tab 3: Platform Analysis
        dbc.Tab(label="Platform Analysis", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Device Usage Distribution", className="card-title"),
                            dcc.Graph(id="device-distribution")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Platform Conversion Rates", className="card-title"),
                            dcc.Graph(id="platform-conversion")
                        ])
                    ])
                ], width=6),
            ]),
        ]),
        
        # Tab 4: Recommendations
        dbc.Tab(label="Recommendations", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("User Lookup", className="card-title"),
                            dcc.Dropdown(
                                id='user-dropdown',
                                options=[{'label': f"User {uid}", 'value': uid} 
                                       for uid in users_df['user_id'].head()],
                                value=users_df['user_id'].iloc[0]
                            ),
                            html.Div(id="user-recommendations")
                        ])
                    ])
                ], width=12),
            ]),
        ]),
    ]),
], fluid=True)

# Callbacks for updating visualizations
@app.callback(
    [Output("total-users", "children"),
     Output("prime-members", "children"),
     Output("total-sales", "children")],
    [Input("user-dropdown", "value")]  # Dummy input to trigger initial load
)
def update_kpi_cards(value):
    total_users = len(users_df)
    prime_members = len(users_df[users_df['is_prime_member']])
    total_sales = f"${events_df[events_df['event_type'] == 'purchase']['final_price'].sum():,.2f}"
    return total_users, prime_members, total_sales

@app.callback(
    Output("sales-trend", "figure"),
    [Input("user-dropdown", "value")]
)
def update_sales_trend(value):
    # Group sales by date
    daily_sales = events_df[events_df['event_type'] == 'purchase'].groupby(
        events_df['timestamp'].dt.date
    )['final_price'].sum().reset_index()
    
    fig = px.line(daily_sales, x='timestamp', y='final_price',
                  title="Daily Sales Trend")
    return fig

@app.callback(
    Output("category-pie", "figure"),
    [Input("user-dropdown", "value")]
)
def update_category_pie(value):
    category_sales = products_df.groupby('category')['base_price'].sum()
    fig = px.pie(values=category_sales.values, names=category_sales.index,
                 title="Sales by Category")
    return fig

@app.callback(
    Output("prime-comparison", "figure"),
    [Input("user-dropdown", "value")]
)
def update_prime_comparison(value):
    prime_comparison = events_df.merge(
        users_df[['user_id', 'is_prime_member']], on='user_id'
    ).groupby('is_prime_member')['final_price'].mean()
    
    fig = px.bar(x=['Non-Prime', 'Prime'], y=prime_comparison.values,
                 title="Average Purchase Value: Prime vs Non-Prime")
    return fig

@app.callback(
    Output("device-distribution", "figure"),
    [Input("user-dropdown", "value")]
)
def update_device_distribution(value):
    device_dist = events_df['device_type'].value_counts()
    fig = px.bar(x=device_dist.index, y=device_dist.values,
                 title="Device Usage Distribution")
    return fig

@app.callback(
    Output("user-recommendations", "children"),
    [Input("user-dropdown", "value")]
)
def update_recommendations(user_id):
    # Simple recommendation based on user's favorite categories
    user_events = events_df[events_df['user_id'] == user_id]
    user_categories = user_events.merge(
        products_df[['product_id', 'category']], on='product_id'
    )['category'].value_counts().head(3)
    
    recommended_products = products_df[
        products_df['category'].isin(user_categories.index)
    ].sample(5)
    
    return dbc.Table.from_dataframe(
        recommended_products[['product_name', 'category', 'base_price']],
        striped=True,
        bordered=True,
        hover=True
    )

if __name__ == '__main__':
    app.run_server(debug=True)