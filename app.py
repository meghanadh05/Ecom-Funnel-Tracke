# app.py
# PROFESSIONAL PORTFOLIO VERSION
# E-commerce Analytics Dashboard with Funnel, Revenue & Cohort Analysis

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Tuple

# ---------------------------#
# Page Configuration
# ---------------------------#
st.set_page_config(
    page_title="Pro E-commerce Analytics Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------#
# Main Dashboard Class
# ---------------------------#

class EcomDashboard:
    """
    A class to encapsulate the logic and UI of the e-commerce dashboard.
    This structure is scalable, maintainable, and reflects professional coding standards.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = self.load_and_prepare_data()

    @st.cache_data
    def load_and_prepare_data(_self) -> pd.DataFrame:
        """
        Loads, cleans, and prepares the e-commerce dataset.
        Using _self as the first arg is a convention for methods cached by Streamlit.
        Returns:
            pd.DataFrame: The prepared DataFrame ready for analysis.
        """
        df = pd.read_csv(_self.data_path)
        
        # Standardize column names
        df.columns = [col.strip().replace(" ", "") for col in df.columns]

        # Data Cleaning and Feature Engineering
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df.dropna(subset=["Timestamp", "UserID", "EventType"], inplace=True)
        df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce').fillna(0)
        df["EventType"] = df["EventType"].astype(str).str.strip().lower()
        
        # Create a 'Month' column for cohort analysis
        df['OrderMonth'] = df['Timestamp'].dt.to_period('M')

        # Map key funnel events
        event_map = {
            "product_view": "Product View", "view": "Product View",
            "add_to_cart": "Add to Cart", "cart": "Add to Cart",
            "purchase": "Purchase"
        }
        df["FunnelEvent"] = df["EventType"].map(event_map)
        
        return df

    def get_filtered_data(self) -> pd.DataFrame:
        """
        Applies sidebar filters to the dataset.
        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        st.sidebar.header("Master Filters")
        
        min_date = self.df["Timestamp"].min().date()
        max_date = self.df["Timestamp"].max().date()

        start_date, end_date = st.sidebar.date_input(
            "Select date range:", (min_date, max_date),
            min_value=min_date, max_value=max_date,
            help="Filter the data to a specific date range for analysis."
        )
        
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        
        return self.df[(self.df['Timestamp'] >= start_ts) & (self.df['Timestamp'] < end_ts)]

    def run(self):
        """
        Main method to run the Streamlit application.
        """
        st.title("🚀 Professional E-commerce Analytics Dashboard")
        st.markdown("""
        This dashboard provides a comprehensive analysis of e-commerce data, including a user-centric sales funnel, 
        revenue trends, and a user retention cohort analysis.
        """)

        df_filtered = self.get_filtered_data()

        if df_filtered.empty:
            st.warning("No data available for the selected date range. Please select a different period.")
            return

        self.render_kpis(df_filtered)
        
        self.render_tabs(df_filtered)

    def render_kpis(self, df: pd.DataFrame):
        """Renders the Key Performance Indicators."""
        
        # --- Accurate User-Centric Calculations ---
        views_users = df[df['FunnelEvent'] == 'Product View']['UserID'].nunique()
        cart_users = df[df['FunnelEvent'] == 'Add to Cart']['UserID'].nunique()
        purchase_users = df[df['FunnelEvent'] == 'Purchase']['UserID'].nunique()
        total_revenue = df[df['FunnelEvent'] == 'Purchase']['Amount'].sum()
        aov = total_revenue / purchase_users if purchase_users > 0 else 0
        overall_conversion = purchase_users / views_users if views_users > 0 else 0

        st.markdown("### 📊 Key Performance Indicators")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue", f"${total_revenue:,.2f}", help="Total revenue from all purchases.")
        kpi2.metric("Avg. Order Value", f"${aov:,.2f}", help="Average revenue per purchase transaction.")
        kpi3.metric("Total Purchases", f"{purchase_users:,}", help="Total number of unique customers who made a purchase.")
        kpi4.metric("Overall Conversion Rate", f"{overall_conversion:.2%}", help="Percentage of users who viewed a product and then made a purchase.")
        st.markdown("---")


    def render_tabs(self, df: pd.DataFrame):
        """Renders the main analysis tabs."""
        tab_funnel, tab_cohort, tab_products = st.tabs([
            "🔽 Funnel Analysis", 
            "👥 Cohort Retention Analysis", 
            "🛒 Product Insights"
        ])

        with tab_funnel:
            self.render_funnel_analysis(df)
        with tab_cohort:
            self.render_cohort_analysis(df)
        with tab_products:
            self.render_product_insights(df)

    def render_funnel_analysis(self, df: pd.DataFrame):
        st.header("Sales Funnel Performance")
        st.markdown("Tracks the number of **unique users** moving through the purchase funnel.")
        
        views = df[df['FunnelEvent'] == 'Product View']['UserID'].nunique()
        carts = df[df['FunnelEvent'] == 'Add to Cart']['UserID'].nunique()
        purchases = df[df['FunnelEvent'] == 'Purchase']['UserID'].nunique()

        fig = go.Figure(go.Funnel(
            y=["Product Views", "Added to Cart", "Purchases"],
            x=[views, carts, purchases],
            textinfo="value + percent initial",
            marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c"]},
        ))
        fig.update_layout(title_text="User Conversion Funnel")
        st.plotly_chart(fig, use_container_width=True)

    def render_cohort_analysis(self, df: pd.DataFrame):
        st.header("User Retention Cohort Analysis")
        st.markdown("""
        This analysis tracks cohorts of users based on their first purchase month and shows what percentage
        of them return to make another purchase in the following months. It's a key indicator of customer loyalty and product-market fit.
        """)
        
        df_purchases = df[df['FunnelEvent'] == 'Purchase'].copy()
        if df_purchases.empty:
            st.info("No purchase data available to generate cohort analysis.")
            return

        df_purchases['CohortMonth'] = df_purchases.groupby('UserID')['OrderMonth'].transform('min')
        
        def get_month_diff(df, event_month_col, cohort_month_col):
            return (df[event_month_col].dt.year - df[cohort_month_col].dt.year) * 12 + \
                   (df[event_month_col].dt.month - df[cohort_month_col].dt.month)

        df_purchases['CohortIndex'] = get_month_diff(df_purchases, df_purchases['OrderMonth'], df_purchases['CohortMonth'])

        cohort_data = df_purchases.groupby(['CohortMonth', 'CohortIndex'])['UserID'].nunique().reset_index()
        cohort_counts = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='UserID')
        
        cohort_sizes = cohort_counts.iloc[:, 0]
        cohort_retention = cohort_counts.divide(cohort_sizes, axis=0)
        cohort_retention.index = cohort_retention.index.strftime('%Y-%m')

        fig = px.imshow(
            cohort_retention, 
            labels=dict(x="Months Since First Purchase", y="First Purchase Month", color="Retention Rate"),
            title="Monthly Customer Retention Rate (%)",
            color_continuous_scale=px.colors.sequential.BuGn
        )
        fig.update_layout(
            xaxis_title="Months Since First Purchase",
            yaxis_title="First Purchase Cohort"
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_product_insights(self, df: pd.DataFrame):
        st.header("Product Performance")
        
        if 'ProductID' not in df.columns:
            st.warning("ProductID column not found. Cannot generate product insights.")
            return

        product_stats = df.groupby('ProductID').agg(
            views=('FunnelEvent', lambda x: (x == 'Product View').sum()),
            purchases=('FunnelEvent', lambda x: (x == 'Purchase').sum()),
            revenue=('Amount', lambda x: x[df.loc[x.index, 'FunnelEvent'] == 'Purchase'].sum())
        ).reset_index()

        product_stats = product_stats[product_stats['views'] > 0]
        product_stats['conversion_rate'] = (product_stats['purchases'] / product_stats['views'])
        
        st.subheader("Top Products by Revenue")
        st.dataframe(
            product_stats.sort_values('revenue', ascending=False).head(15).style.format({
                'revenue': '${:,.2f}', 'conversion_rate': '{:.2%}'
            }),
            use_container_width=True
        )

# ---------------------------#
# App Execution
# ---------------------------#

if __name__ == "__main__":
    try:
        dashboard = EcomDashboard(data_path="ecommerce_clickstream_transactions.csv")
        dashboard.run()
    except FileNotFoundError:
        st.error("The data file ('ecommerce_clickstream_transactions.csv') was not found. Please make sure it's in the same directory as the app.py script.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
