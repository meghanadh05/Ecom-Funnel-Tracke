# app.py
# ADVANCED E-commerce Funnel & Revenue Dashboard
# Provides accurate, user-centric conversion metrics and deep product/revenue insights.

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date

# ---------------------------#
# Page Config
# ---------------------------#
st.set_page_config(
    page_title="Advanced E-commerce Dashboard",
    page_icon="💰",
    layout="wide"
)

# ---------------------------#
# Helper Functions & Data Loading
# ---------------------------#

@st.cache_data
def load_and_prep_data(path: str) -> pd.DataFrame:
    """Loads, standardizes, and prepares the data for analysis."""
    df = pd.read_csv(path)
    
    # Standardize column names for robustness
    rename_map = {c: c.strip().replace(" ", "") for c in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure essential columns exist
    for col in ["UserID", "Timestamp", "EventType"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Data cleaning and typing
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp", "UserID", "EventType"], inplace=True)
    df["EventType"] = df["EventType"].astype(str).str.strip().str.lower()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)

    # Standardize main funnel events
    event_map = {
        "product_view": "Product View", "view": "Product View",
        "add_to_cart": "Add to Cart", "cart": "Add to Cart",
        "purchase": "Purchase"
    }
    df["EventTypeMapped"] = df["EventType"].map(event_map)
    
    return df

# ---------------------------#
# Main App Logic
# ---------------------------#

# --- Data Loading ---
try:
    df_raw = load_and_prep_data("ecommerce_clickstream_transactions.csv")
    df_funnel_base = df_raw[df_raw["EventTypeMapped"].notna()].copy()
except Exception as e:
    st.error(f"❌ **Error Loading Data:** {e}")
    st.info("Please ensure 'ecommerce_clickstream_transactions.csv' is in the same folder as this script.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Date Range Filter")
min_date = df_funnel_base["Timestamp"].min().date()
max_date = df_funnel_base["Timestamp"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Select date range:",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply date filter
start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
df_funnel = df_funnel_base[(df_funnel_base['Timestamp'] >= start_ts) & (df_funnel_base['Timestamp'] < end_ts)]


# --- Core Metrics Calculation (ACCURATE: Based on Unique Users) ---
if not df_funnel.empty:
    views_users = df_funnel[df_funnel['EventTypeMapped'] == 'Product View']['UserID'].nunique()
    cart_users = df_funnel[df_funnel['EventTypeMapped'] == 'Add to Cart']['UserID'].nunique()
    purchase_users = df_funnel[df_funnel['EventTypeMapped'] == 'Purchase']['UserID'].nunique()
    
    # Revenue Metrics
    total_revenue = df_funnel[df_funnel['EventTypeMapped'] == 'Purchase']['Amount'].sum()
    aov = total_revenue / purchase_users if purchase_users > 0 else 0
    
    # Conversion Rates
    view_to_cart_rate = cart_users / views_users if views_users > 0 else 0
    cart_to_purchase_rate = purchase_users / cart_users if cart_users > 0 else 0
    overall_conversion_rate = purchase_users / views_users if views_users > 0 else 0
else: # Handle empty dataframe after filtering
    views_users, cart_users, purchase_users, total_revenue, aov = 0, 0, 0, 0, 0
    view_to_cart_rate, cart_to_purchase_rate, overall_conversion_rate = 0, 0, 0

# --- Dashboard UI ---
st.title("💰 Advanced E-commerce Dashboard")
st.markdown("An accurate, user-centric analysis of your sales funnel and revenue.")

# --- KPIs ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Revenue", f"${total_revenue:,.2f}")
kpi2.metric("Average Order Value (AOV)", f"${aov:,.2f}")
kpi3.metric("Total Purchases", f"{purchase_users:,}")
kpi4.metric("Overall Conversion Rate", f"{overall_conversion_rate:.2%}")

st.markdown("---")


# --- Tabs for Deeper Analysis ---
tab_funnel, tab_revenue, tab_products = st.tabs(["🔽 Funnel Analysis", "📈 Revenue Trends", "🛒 Product Performance"])

with tab_funnel:
    st.header("Sales Funnel Performance")
    st.markdown("This funnel tracks the number of **unique users** who performed each action.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Funnel Chart
        fig = go.Figure(go.Funnel(
            y=["Product Views", "Added to Cart", "Purchases"],
            x=[views_users, cart_users, purchase_users],
            textinfo="value + percent initial + percent previous",
            marker={"color": ["#0099ff", "#ff9900", "#33cc33"]},
        ))
        fig.update_layout(title_text="User Conversion Funnel")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Funnel Metrics Table
        st.subheader("Funnel Breakdown")
        funnel_data = {
            "Stage": ["Product View", "Add to Cart", "Purchase"],
            "Unique Users": [f"{views_users:,}", f"{cart_users:,}", f"{purchase_users:,}"],
            "Conversion from Previous": ["-", f"{view_to_cart_rate:.2%}", f"{cart_to_purchase_rate:.2%}"],
            "Overall Conversion": ["-", "-", f"{overall_conversion_rate:.2%}"]
        }
        st.table(pd.DataFrame(funnel_data))
        st.markdown(f"""
        **Key Insights:**
        - **Drop-off (View → Cart):** `{1-view_to_cart_rate:.2%}` of users viewed products but didn't add to cart.
        - **Drop-off (Cart → Purchase):** `{1-cart_to_purchase_rate:.2%}` of users added items to their cart but did not complete the purchase.
        """)

with tab_revenue:
    st.header("Revenue & Purchase Trends")
    
    # Daily Revenue Chart
    df_purchases = df_funnel[df_funnel['EventTypeMapped'] == 'Purchase'].copy()
    if not df_purchases.empty:
        daily_revenue = df_purchases.set_index('Timestamp').resample('D')['Amount'].sum().reset_index()
        fig_rev = px.line(daily_revenue, x='Timestamp', y='Amount', title='Daily Revenue Over Time', markers=True,
                          labels={'Timestamp': 'Date', 'Amount': 'Total Revenue ($)'})
        fig_rev.update_layout(hovermode="x unified")
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("No purchase data available for the selected date range to display revenue trends.")

with tab_products:
    st.header("Product Performance Analysis")
    st.markdown("Analyze which products convert best and generate the most revenue.")

    if 'ProductID' not in df_funnel.columns:
        st.warning("ProductID column not found. Cannot perform product analysis.")
    else:
        # Product Performance Calculation
        product_stats = df_funnel.groupby('ProductID').agg(
            views=('EventTypeMapped', lambda x: (x == 'Product View').sum()),
            carts=('EventTypeMapped', lambda x: (x == 'Add to Cart').sum()),
            purchases=('EventTypeMapped', lambda x: (x == 'Purchase').sum()),
            revenue=('Amount', lambda x: x[df_funnel.loc[x.index, 'EventTypeMapped'] == 'Purchase'].sum())
        ).reset_index()

        product_stats = product_stats[(product_stats['views'] > 0) | (product_stats['purchases'] > 0)]
        product_stats['view_to_cart_rate'] = (product_stats['carts'] / product_stats['views']).fillna(0)
        product_stats['cart_to_purchase_rate'] = (product_stats['purchases'] / product_stats['carts']).fillna(0)
        product_stats['overall_conversion'] = (product_stats['purchases'] / product_stats['views']).fillna(0)
        
        st.subheader("Top Products by Revenue")
        st.dataframe(
            product_stats.sort_values('revenue', ascending=False).head(20).style.format({
                'revenue': '${:,.2f}',
                'view_to_cart_rate': '{:.2%}',
                'cart_to_purchase_rate': '{:.2%}',
                'overall_conversion': '{:.2%}'
            }),
            use_container_width=True
        )

        st.subheader("Products with Highest Conversion Rate (View -> Purchase)")
        st.dataframe(
            product_stats[product_stats['views'] > 10].sort_values('overall_conversion', ascending=False).head(20).style.format({
                'revenue': '${:,.2f}',
                'view_to_cart_rate': '{:.2%}',
                'cart_to_purchase_rate': '{:.2%}',
                'overall_conversion': '{:.2%}'
            }),
            use_container_width=True
        )
