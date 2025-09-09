# app.py
# COMPREHENSIVE E-COMMERCE ANALYTICS DASHBOARD (PORTFOLIO EDITION)
# Features: Deep EDA, Funnel, Temporal, Customer & Product Analysis

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------#
# Page Configuration
# ---------------------------#
st.set_page_config(
    page_title="Comprehensive E-commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------#
# Main Dashboard Class
# ---------------------------#

class EcomDashboard:
    """
    Encapsulates the entire dashboard logic for a clean, scalable, and professional structure.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = self.load_and_prepare_data()

    @st.cache_data
    def load_and_prepare_data(_self) -> pd.DataFrame:
        """Loads, cleans, and prepares the dataset."""
        df = pd.read_csv(_self.data_path)
        df.columns = [col.strip().replace(" ", "") for col in df.columns]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df.dropna(subset=["Timestamp", "UserID", "EventType"], inplace=True)
        df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce').fillna(0)
        df["EventType"] = df["EventType"].astype(str).str.strip().lower()
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        event_map = {"product_view": "Product View", "add_to_cart": "Add to Cart", "purchase": "Purchase"}
        df["FunnelEvent"] = df["EventType"].map(event_map)
        return df

    def run(self):
        """Main method to render the Streamlit application."""
        st.title("📊 Comprehensive E-commerce Analytics Dashboard")
        
        st.sidebar.title("Filters")
        start_date, end_date = st.sidebar.date_input(
            "Select date range:",
            (self.df["Timestamp"].min().date(), self.df["Timestamp"].max().date()),
            min_value=self.df["Timestamp"].min().date(),
            max_value=self.df["Timestamp"].max().date()
        )
        df_filtered = self.df[
            (self.df['Timestamp'].dt.date >= start_date) & 
            (self.df['Timestamp'].dt.date <= end_date)
        ]

        if df_filtered.empty:
            st.warning("No data in selected date range.")
            return

        tabs = st.tabs([
            "📖 Data Overview (EDA)", "🔽 Funnel Analysis", "⏰ Temporal Analysis", 
            "👥 Customer Insights", "🛒 Product Insights"
        ])
        with tabs[0]: self.render_eda_tab(df_filtered)
        with tabs[1]: self.render_funnel_tab(df_filtered)
        with tabs[2]: self.render_temporal_tab(df_filtered)
        with tabs[3]: self.render_customer_tab(df_filtered)
        with tabs[4]: self.render_product_tab(df_filtered)

    def render_eda_tab(self, df: pd.DataFrame):
        st.header("Exploratory Data Analysis (EDA)")
        st.markdown("A high-level overview of the dataset's structure and contents.")

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Total Columns", f"{len(df.columns)}")
        col3.metric("Date Range Start", df['Timestamp'].min().strftime('%Y-%m-%d'))
        col4.metric("Date Range End", df['Timestamp'].max().strftime('%Y-%m-%d'))
        st.markdown("---")

        # Data Preview & Schema
        c1, c2 = st.columns((1, 1))
        with c1:
            st.subheader("Data Preview")
            st.dataframe(df.head())
        with c2:
            st.subheader("Dataset Schema")
            schema_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(t) for t in df.dtypes],
                'Non-Null Values': df.count().values
            })
            st.dataframe(schema_df)

        st.markdown("---")

        # Event Type Distribution
        st.subheader("Distribution of All Event Types")
        event_counts = df['EventType'].value_counts()
        fig = px.pie(event_counts, values=event_counts.values, names=event_counts.index, 
                     title="Proportion of All Recorded Events", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    def render_funnel_tab(self, df: pd.DataFrame):
        st.header("User-Centric Sales Funnel")
        st.markdown("Tracks the number of **unique users** moving through the purchase funnel.")
        
        views = df[df['FunnelEvent'] == 'Product View']['UserID'].nunique()
        carts = df[df['FunnelEvent'] == 'Add to Cart']['UserID'].nunique()
        purchases = df[df['FunnelEvent'] == 'Purchase']['UserID'].nunique()

        fig = go.Figure(go.Funnel(
            y=["Product Views", "Added to Cart", "Purchases"],
            x=[views, carts, purchases],
            textinfo="value + percent initial + percent previous",
            marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c"]}
        ))
        st.plotly_chart(fig, use_container_width=True)

    def render_temporal_tab(self, df: pd.DataFrame):
        st.header("Temporal (Time-Based) Analysis")

        # Activity Heatmap
        st.subheader("User Activity by Day and Hour")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        activity = df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='counts')
        activity['DayOfWeek'] = pd.Categorical(activity['DayOfWeek'], categories=day_order, ordered=True)
        fig = px.density_heatmap(activity, x='Hour', y='DayOfWeek', z='counts', title="Activity Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily Event Trends
        st.subheader("Daily Funnel Activity")
        daily_counts = df[df['FunnelEvent'].notna()].groupby([df['Timestamp'].dt.date, 'FunnelEvent']).size().reset_index(name='count')
        fig2 = px.line(daily_counts, x='Timestamp', y='count', color='FunnelEvent', title="Daily Funnel Events Over Time")
        st.plotly_chart(fig2, use_container_width=True)

    def render_customer_tab(self, df: pd.DataFrame):
        st.header("Customer Insights")
        
        # User Leaderboard
        st.subheader("Top Users Leaderboard")
        user_metrics = df.groupby('UserID').agg(
            total_events=('EventType', 'count'),
            total_purchases=('FunnelEvent', lambda x: (x == 'Purchase').sum()),
            total_revenue=('Amount', 'sum')
        ).sort_values('total_revenue', ascending=False).reset_index()
        st.dataframe(user_metrics.head(10).style.format({'total_revenue': '${:,.2f}'}))

    def render_product_tab(self, df: pd.DataFrame):
        st.header("Product Performance Insights")
        
        if 'ProductID' not in df.columns:
            st.warning("ProductID column not found.")
            return

        product_stats = df[df['FunnelEvent'].notna()].groupby('ProductID').agg(
            views=('FunnelEvent', lambda x: (x == 'Product View').sum()),
            carts=('FunnelEvent', lambda x: (x == 'Add to Cart').sum()),
            purchases=('FunnelEvent', lambda x: (x == 'Purchase').sum()),
            revenue=('Amount', lambda x: x[df.loc[x.index, 'FunnelEvent'] == 'Purchase'].sum())
        ).reset_index()

        product_stats = product_stats[product_stats['views'] > 0]
        product_stats['conversion_rate'] = (product_stats['purchases'] / product_stats['views'])

        st.subheader("Product Conversion Funnel")
        st.markdown("Identify products that are popular (high views) but fail to convert, and vice-versa.")
        fig = px.scatter(
            product_stats.sort_values('revenue', ascending=False).head(50), 
            x='views', y='conversion_rate', size='revenue', color='revenue',
            hover_name='ProductID',
            labels={'views': 'Number of Views', 'conversion_rate': 'Conversion Rate (%)'},
            title='Product Performance: Views vs. Conversion'
        )
        fig.update_layout(yaxis=dict(tickformat=".2%"))
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    try:
        dashboard = EcomDashboard(data_path="ecommerce_clickstream_transactions.csv")
        dashboard.run()
    except FileNotFoundError:
        st.error("Data file not found. Ensure 'ecommerce_clickstream_transactions.csv' is in the same folder.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
