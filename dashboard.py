import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import uuid
import random
from faker import Faker

# Initialize Faker for generating realistic data
fake = Faker()

# Amazon color scheme
AMAZON_COLORS = {
    'primary': '#232F3E',     # Dark blue
    'secondary': '#FF9900',   # Orange
    'accent1': '#146EB4',     # Light blue
    'accent2': '#232F3E',     # Navy
    'success': '#067D62',     # Green
    'warning': '#FF9900',     # Orange
    'background': '#EAEDED',  # Light gray
    'text': '#111111'         # Almost black
}

class AmazonDataGenerator:
    def __init__(self, start_date='2023-01-01', end_date='2024-12-31'):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        self.categories = {
            'Electronics': ['Amazon Devices', 'Phones', 'Computers', 'Smart Home'],
            'Fashion': ['Clothing', 'Shoes', 'Jewelry', 'Watches'],
            'Home': ['Kitchen', 'Furniture', 'Decor', 'Garden'],
            'Media': ['Books', 'Kindle', 'Audible', 'Prime Video'],
            'Grocery': ['Fresh', 'Pantry', 'Whole Foods', 'Snacks']
        }
        
    def generate_products(self, num_products=1000):
        products = []
        for _ in range(num_products):
            category = random.choice(list(self.categories.keys()))
            subcategory = random.choice(self.categories[category])
            
            product = {
                'product_id': str(uuid.uuid4()),
                'name': f"{fake.company()} {fake.word().title()}",
                'category': category,
                'subcategory': subcategory,
                'price': round(random.uniform(10, 1000), 2),
                'rating': round(random.uniform(3.5, 5), 1),
                'prime_eligible': random.random() < 0.8,
                'stock_level': random.randint(0, 1000),
                'discount_percentage': random.choice([0, 0, 0, 10, 15, 20, 25, 30]),
                'seller_rating': round(random.uniform(3.8, 5), 1),
                'review_count': random.randint(0, 10000)
            }
            products.append(product)
        return pd.DataFrame(products)

    def generate_users(self, num_users=1000):
        users = []
        for _ in range(num_users):
            is_prime = random.random() < 0.6
            user = {
                'user_id': str(uuid.uuid4()),
                'join_date': fake.date_between(
                    start_date=self.start_date,
                    end_date=self.end_date
                ),
                'is_prime': is_prime,
                'age': random.randint(18, 80),
                'preferred_category': random.choice(list(self.categories.keys())),
                'preferred_device': random.choice(['mobile', 'desktop', 'tablet', 'alexa']),
                'country': fake.country(),
                'total_spend': round(random.uniform(0, 5000), 2),
                'avg_order_value': round(random.uniform(20, 200), 2),
                'purchase_frequency': random.choice(['High', 'Medium', 'Low']),
                'last_purchase_date': fake.date_between(
                    start_date=self.start_date,
                    end_date=self.end_date
                )
            }
            users.append(user)
        return pd.DataFrame(users)

    def generate_events(self, products_df, users_df, num_events=10000):
        events = []
        event_types = ['view', 'cart', 'purchase', 'remove_from_cart', 'wishlist', 'review']
        devices = ['mobile', 'desktop', 'tablet', 'alexa']
        
        for _ in range(num_events):
            user = users_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]
            event_type = random.choices(
                event_types, 
                weights=[0.5, 0.2, 0.15, 0.05, 0.05, 0.05]
            )[0]
            
            # Generate timestamp with realistic patterns
            hour = random.choices(
                range(24),
                weights=[2,1,1,1,1,2,3,5,7,8,7,6,7,8,7,6,7,8,7,6,5,4,3,2]
            )[0]
            
            timestamp = fake.date_time_between(
                start_date=self.start_date,
                end_date=self.end_date
            ).replace(hour=hour)
            
            event = {
                'event_id': str(uuid.uuid4()),
                'user_id': user['user_id'],
                'product_id': product['product_id'],
                'event_type': event_type,
                'timestamp': timestamp,
                'device': random.choice(devices),
                'price': product['price'],
                'category': product['category'],
                'subcategory': product['subcategory'],
                'session_duration': random.randint(10, 3600),
                'page_views': random.randint(1, 20),
                'is_prime_user': user['is_prime'],
                'user_age': user['age'],
                'discount_applied': random.random() < 0.3,
                'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Amazon Pay']),
                'shipping_method': random.choice(['Standard', 'Prime', 'Express', 'Next Day'])
            }
            events.append(event)
        return pd.DataFrame(events)

# Generate data
generator = AmazonDataGenerator()
products_df = generator.generate_products(1000)
users_df = generator.generate_users(500)
events_df = generator.generate_events(products_df, users_df, 10000)

# Data preprocessing
events_df['hour'] = events_df['timestamp'].dt.hour
events_df['day_of_week'] = events_df['timestamp'].dt.day_name()
events_df['month'] = events_df['timestamp'].dt.month_name()
events_df['week'] = events_df['timestamp'].dt.isocalendar().week
events_df['quarter'] = events_df['timestamp'].dt.quarter

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Amazon Analytics Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: ''' + AMAZON_COLORS['background'] + ''';
                font-family: "Amazon Ember", Arial, sans-serif;
            }
            .nav-link.active {
                background-color: ''' + AMAZON_COLORS['secondary'] + ''' !important;
                color: white !important;
            }
            .card {
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: none;
                margin-bottom: 1rem;
            }
            .card-header {
                background-color: ''' + AMAZON_COLORS['primary'] + ''';
                color: white;
                font-weight: 500;
            }
            .dashboard-title {
                color: ''' + AMAZON_COLORS['primary'] + ''';
                font-weight: bold;
                padding: 1rem 0;
            }
            .kpi-card {
                text-align: center;
                padding: 1rem;
            }
            .kpi-value {
                font-size: 2rem;
                font-weight: bold;
                color: ''' + AMAZON_COLORS['secondary'] + ''';
            }
            .filter-label {
                font-weight: 500;
                margin-bottom: 0.5rem;
            }
            .dash-graph {
                background-color: white;
                border-radius: 4px;
                padding: 1rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout components
def create_header():
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/amazon-logo.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("Analytics Dashboard", className="ms-2")),
                ],
                align="center",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Overview", href="#", active=True)),
                    dbc.NavItem(dbc.NavLink("Reports", href="#")),
                    dbc.NavItem(dbc.NavLink("Settings", href="#")),
                ]),
                id="navbar-collapse",
                navbar=True,
            ),
        ]),
        color=AMAZON_COLORS['primary'],
        dark=True,
        className="mb-4",
    )

def create_filters():
    return dbc.Card([
        dbc.CardHeader("Filters & Controls"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Date Range", className="filter-label"),
                    dcc.DatePickerRange(
                        id='date-filter',
                        start_date=events_df['timestamp'].min(),
                        end_date=events_df['timestamp'].max(),
                        className="mb-3"
                    )
                ], md=3),
                
                dbc.Col([
                    html.Label("Category", className="filter-label"),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': cat, 'value': cat} 
                               for cat in products_df['category'].unique()],
                        multi=True,
                        placeholder="Select categories..."
                    )
                ], md=3),
                
                dbc.Col([
                    html.Label("Device", className="filter-label"),
                    dcc.Dropdown(
                        id='device-filter',
                        options=[{'label': dev.title(), 'value': dev} 
                               for dev in events_df['device'].unique()],
                        multi=True,
                        placeholder="Select devices..."
                    )
                ], md=3),
                
                dbc.Col([
                    html.Label("Price Range ($)", className="filter-label"),
                    dcc.RangeSlider(
                        id='price-range',
                        min=0,
                        max=1000,
                        step=50,
                        marks={i: f'${i}' for i in range(0, 1001, 200)},
                        value=[0, 1000]
                    )
                ], md=3),
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Time of Day", className="filter-label"),
                    dcc.RangeSlider(
                        id='hour-range',
                        min=0,
                        max=23,
                        step=1,
                        marks={i: f'{i:02d}:00' for i in range(0, 24, 4)},
                        value=[0, 23]
                    )
                ], md=6),
                
                dbc.Col([
                    html.Label("User Type", className="filter-label"),
                    dbc.ButtonGroup([
                        dbc.Button(
                            "All Users",
                            id="all-users",
                            color="primary",
                            active=True,
                            className="me-1"
                        ),
                        dbc.Button(
                            "Prime",
                            id="prime-users-btn",
                            color="primary",
                            className="me-1"
                        ),
                        dbc.Button(
                            "Non-Prime",
                            id="non-prime-users-btn",
                            color="primary"
                        ),
                    ])
                ], md=6),
            ], className="mt-3"),
        ])
    ], className="mb-4")

def create_kpi_cards():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Sales", className="card-title"),
                    html.H2(id="total-sales", className="text-primary"),
                    dbc.Progress(id="sales-progress", value=75, color=AMAZON_COLORS['secondary']),
                    html.Small(id="sales-trend", className="text-muted")
                ])
            ], className="kpi-card")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Conversion Rate", className="card-title"),
                    html.H2(id="conversion-rate", className="text-success"),
                    dbc.Progress(id="conversion-progress", value=65, color=AMAZON_COLORS['success']),
                    html.Small(id="conversion-trend", className="text-muted")
                ])
            ], className="kpi-card")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Prime Users", className="card-title"),
                    html.H2(id="prime-users", className="text-info"),
                    dbc.Progress(id="prime-progress", value=60, color=AMAZON_COLORS['accent1']),
                    html.Small(id="prime-trend", className="text-muted")
                ])
            ], className="kpi-card")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Avg Order Value", className="card-title"),
                    html.H2(id="avg-order", className="text-warning"),
                    dbc.Progress(id="aov-progress", value=70, color=AMAZON_COLORS['warning']),
html.Small(id="aov-trend", className="text-muted")
                ])
            ], className="kpi-card")
        ], width=3),
    ], className="mb-4")

def create_charts_section():
    return dbc.Tabs([
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sales Trend"),
                        dbc.CardBody([
                            dcc.Graph(id="sales-trend-graph")
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Category Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="category-pie")
                        ])
                    ])
                ], width=4),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Hourly Traffic Pattern"),
                        dbc.CardBody([
                            dcc.Graph(id="hourly-traffic")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Customer Journey Funnel"),
                        dbc.CardBody([
                            dcc.Graph(id="customer-funnel")
                        ])
                    ])
                ], width=6),
            ]),
        ], label="Overview"),
        
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("Cohort Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="cohort-heatmap")
                ])
            ])
        ], label="Cohort Analysis"),
        
        dbc.Tab([
            dbc.Card([
                dbc.CardHeader("User Behavior"),
                dbc.CardBody([
                    dcc.Graph(id="user-behavior-sankey")
                ])
            ])
        ], label="User Behavior"),
    ], className="mb-4")

# Main layout
app.layout = dbc.Container([
    create_header(),
    html.H1("Amazon Analytics Dashboard", className="dashboard-title"),
    create_filters(),
    create_kpi_cards(),
    create_charts_section(),
    
    # Footer
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Small(
                        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        className="text-muted"
                    )
                ])
            ])
        ])
    ])
], fluid=True)

# Callback functions
@app.callback(
    [Output("total-sales", "children"),
     Output("conversion-rate", "children"),
     Output("prime-users", "children"),
     Output("avg-order", "children"),
     Output("sales-trend", "children"),
     Output("conversion-trend", "children"),
     Output("prime-trend", "children"),
     Output("aov-trend", "children")],
    [Input("date-filter", "start_date"),
     Input("date-filter", "end_date"),
     Input("category-filter", "value"),
     Input("device-filter", "value"),
     Input("price-range", "value"),
     Input("hour-range", "value")]
)
def update_kpis(start_date, end_date, categories, devices, price_range, hour_range):
    filtered_df = apply_filters(events_df, start_date, end_date, categories, 
                              devices, price_range, hour_range)
    
    # Calculate KPIs
    total_sales = calculate_total_sales(filtered_df)
    conversion_rate = calculate_conversion_rate(filtered_df)
    prime_percentage = calculate_prime_percentage(filtered_df)
    avg_order_value = calculate_avg_order_value(filtered_df)
    
    # Calculate trends
    sales_trend = calculate_trend(filtered_df, 'sales')
    conversion_trend = calculate_trend(filtered_df, 'conversion')
    prime_trend = calculate_trend(filtered_df, 'prime')
    aov_trend = calculate_trend(filtered_df, 'aov')
    
    return (
        f"${total_sales:,.2f}",
        f"{conversion_rate:.1f}%",
        f"{prime_percentage:.1f}%",
        f"${avg_order_value:.2f}",
        sales_trend,
        conversion_trend,
        prime_trend,
        aov_trend
    )

@app.callback(
    [Output("sales-trend-graph", "figure"),
     Output("category-pie", "figure"),
     Output("hourly-traffic", "figure"),
     Output("customer-funnel", "figure"),
     Output("cohort-heatmap", "figure"),
     Output("user-behavior-sankey", "figure")],
    [Input("date-filter", "start_date"),
     Input("date-filter", "end_date"),
     Input("category-filter", "value"),
     Input("device-filter", "value"),
     Input("price-range", "value"),
     Input("hour-range", "value")]
)
def update_charts(start_date, end_date, categories, devices, price_range, hour_range):
    filtered_df = apply_filters(events_df, start_date, end_date, categories, 
                              devices, price_range, hour_range)
    
    sales_trend = create_sales_trend_figure(filtered_df)
    category_dist = create_category_distribution_figure(filtered_df)
    hourly_traffic = create_hourly_traffic_figure(filtered_df)
    customer_funnel = create_customer_funnel_figure(filtered_df)
    cohort_heatmap = create_cohort_heatmap_figure(filtered_df)
    user_behavior = create_user_behavior_sankey_figure(filtered_df)
    
    return (
        sales_trend,
        category_dist,
        hourly_traffic,
        customer_funnel,
        cohort_heatmap,
        user_behavior
    )

# Helper functions for data filtering and calculations
def apply_filters(df, start_date, end_date, categories, devices, price_range, hour_range):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['timestamp'] >= start_date) &
            (filtered_df['timestamp'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    if devices:
        filtered_df = filtered_df[filtered_df['device'].isin(devices)]
    
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_range[0]) &
        (filtered_df['price'] <= price_range[1]) &
        (filtered_df['hour'] >= hour_range[0]) &
        (filtered_df['hour'] <= hour_range[1])
    ]
    
    return filtered_df

def calculate_total_sales(df):
    return df[df['event_type'] == 'purchase']['price'].sum()

def calculate_conversion_rate(df):
    views = len(df[df['event_type'] == 'view'])
    purchases = len(df[df['event_type'] == 'purchase'])
    return (purchases / views * 100) if views > 0 else 0

def calculate_prime_percentage(df):
    return df['is_prime_user'].mean() * 100

def calculate_avg_order_value(df):
    purchases = df[df['event_type'] == 'purchase']
    return purchases['price'].mean() if len(purchases) > 0 else 0

def calculate_trend(df, metric_type):
    # Calculate period-over-period change
    current_value = calculate_metric_value(df, metric_type, period='current')
    previous_value = calculate_metric_value(df, metric_type, period='previous')
    
    if previous_value == 0:
        return "No previous data"
    
    change = ((current_value - previous_value) / previous_value) * 100
    arrow = "↑" if change > 0 else "↓"
    color = "text-success" if change > 0 else "text-danger"
    
    return html.Span(
        f"{arrow} {abs(change):.1f}% vs previous period",
        className=color
    )

# Chart creation functions
def create_sales_trend_figure(df):
    daily_sales = df[df['event_type'] == 'purchase'].groupby(
        df['timestamp'].dt.date
    )['price'].sum().reset_index()
    
    fig = px.line(
        daily_sales,
        x='timestamp',
        y='price',
        title="Daily Sales Trend"
    )
    
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=AMAZON_COLORS['background'],
        paper_bgcolor='white',
        font={'color': AMAZON_COLORS['text']},
        title_font_color=AMAZON_COLORS['primary'],
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Sales ($)"
    )
    
    return fig

def create_category_distribution_figure(df):
    category_sales = df[df['event_type'] == 'purchase'].groupby('category')['price'].sum()
    
    fig = px.pie(
        values=category_sales.values,
        names=category_sales.index,
        title="Sales by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=AMAZON_COLORS['background'],
        paper_bgcolor='white'
    )
    
    return fig

def create_hourly_traffic_figure(df):
    hourly_traffic = df.groupby('hour').size().reset_index(name='count')
    
    fig = px.bar(
        hourly_traffic,
        x='hour',
        y='count',
        title="Hourly Traffic Pattern"
    )
    
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=AMAZON_COLORS['background'],
        paper_bgcolor='white',
        xaxis_title="Hour of Day",
        yaxis_title="Number of Events"
    )
    
    return fig

def create_customer_funnel_figure(df):
    funnel_data = df['event_type'].value_counts()
    
    fig = go.Figure(go.Funnel(
        y=['View', 'Add to Cart', 'Purchase'],
        x=[
            funnel_data.get('view', 0),
            funnel_data.get('cart', 0),
            funnel_data.get('purchase', 0)
        ]
    ))
    
    fig.update_layout(
        title="Customer Journey Funnel",
        template="plotly_white"
    )
    
    return fig

def create_cohort_heatmap_figure(df):
    # Create cohort analysis matrix
    cohort_data = calculate_cohort_data(df)
    
    fig = px.imshow(
        cohort_data,
        title="Customer Cohort Analysis",
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Months Since First Purchase",
        yaxis_title="Cohort"
    )
    
    return fig

def create_user_behavior_sankey_figure(df):
    # Create Sankey diagram data
    nodes, links = calculate_user_journey_data(df)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue"
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value']
        )
    )])
    
    fig.update_layout(
        title="User Behavior Flow",
        template="plotly_white"
    )
    
    return fig

# Additional helper functions
def calculate_metric_value(df, metric_type, period='current'):
    if metric_type == 'sales':
        return calculate_total_sales(df)
    elif metric_type == 'conversion':
        return calculate_conversion_rate(df)
    elif metric_type == 'prime':
        return calculate_prime_percentage(df)
    elif metric_type == 'aov':
        return calculate_avg_order_value(df)

def calculate_cohort_data(df):
    # Implement cohort analysis logic
    # This is a placeholder - implement actual cohort calculation
    return pd.DataFrame(np.random.rand(6, 6))

def calculate_user_journey_data(df):
    # Implement user journey calculation logic
    # This is a placeholder - implement actual journey calculation
    nodes = ['View', 'Cart', 'Purchase']
    links = {
        'source': [0, 0, 1],
        'target': [1, 2, 2],
        'value': [100, 50, 30]
    }
    return nodes, links

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=10000, debug=True)
