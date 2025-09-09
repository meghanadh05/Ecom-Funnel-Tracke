import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="E-commerce Funnel Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("ecommerce_clickstream_transactions.csv")

df = load_data()
df.dropna(inplace=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# --- Event Mapping ---
event_name_map = {
    'product_view': 'Product View',
    'add_to_cart': 'Add to Cart',
    'purchase': 'Purchase'
}
df['EventType'] = df['EventType'].map(event_name_map)
main_funnel_events = ['Product View', 'Add to Cart', 'Purchase']
df_funnel = df[df['EventType'].isin(main_funnel_events)]

# --- Sidebar Filters ---
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['Timestamp'].min(), df['Timestamp'].max()]
)
df_funnel = df_funnel[
    (df_funnel['Timestamp'].dt.date >= date_range[0]) &
    (df_funnel['Timestamp'].dt.date <= date_range[1])
]

if "Device" in df.columns:
    device_filter = st.sidebar.multiselect("Device", df['Device'].unique(), default=df['Device'].unique())
    df_funnel = df_funnel[df_funnel['Device'].isin(device_filter)]

# --- Tabs for Navigation ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔽 Funnel", "📈 Trends", "🛒 Product Insights"])

# --- Tab 1: Overview ---
with tab1:
    st.title("E-commerce Funnel Dashboard 🛍️")

    funnel_counts = df_funnel['EventType'].value_counts().reindex(main_funnel_events).fillna(0)
    total_views = int(funnel_counts['Product View'])
    total_cart = int(funnel_counts['Add to Cart'])
    total_purchase = int(funnel_counts['Purchase'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Views", total_views)
    col2.metric("Added to Cart", total_cart, f"{(total_cart/total_views*100):.1f}%" if total_views else "0%")
    col3.metric("Purchases", total_purchase, f"{(total_purchase/total_cart*100):.1f}%" if total_cart else "0%")
    col4.metric("Overall Conversion", f"{(total_purchase/total_views*100):.1f}%" if total_views else "0%")

    # Pie chart of distribution
    st.subheader("Event Distribution")
    pie_fig = px.pie(
        df_funnel,
        names="EventType",
        title="Share of Funnel Events"
    )
    st.plotly_chart(pie_fig, use_container_width=True)

# --- Tab 2: Funnel ---
with tab2:
    st.subheader("Funnel Overview")
    funnel_fig = go.Figure(go.Funnel(
        y=funnel_counts.index,
        x=funnel_counts.values,
        textinfo="value+percent previous+percent initial"
    ))
    st.plotly_chart(funnel_fig, use_container_width=True)

# --- Tab 3: Trends ---
with tab3:
    st.subheader("Daily Funnel Trends")
    funnel_time = df_funnel.groupby([pd.Grouper(key="Timestamp", freq="D"), "EventType"]).size().reset_index(name="Count")
    trend_fig = px.line(
        funnel_time,
        x="Timestamp",
        y="Count",
        color="EventType",
        markers=True
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    # Heatmap for activity
    st.subheader("User Activity Heatmap")
    df_funnel["Day"] = df_funnel["Timestamp"].dt.day_name()
    df_funnel["Hour"] = df_funnel["Timestamp"].dt.hour
    heatmap_data = df_funnel.groupby(["Day", "Hour"]).size().reset_index(name="Count")
    heatmap_fig = px.density_heatmap(
        heatmap_data,
        x="Hour",
        y="Day",
        z="Count",
        color_continuous_scale="Blues",
        title="Activity by Day and Hour"
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

# --- Tab 4: Product Insights ---
with tab4:
    if "ProductID" in df_funnel.columns:
        st.subheader("Top Products by Purchases")
        top_products = df_funnel[df_funnel['EventType'] == "Purchase"]['ProductID'].value_counts().head(10)
        bar_fig = px.bar(
            top_products,
            x=top_products.index,
            y=top_products.values,
            title="Top 10 Purchased Products"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    if "Category" in df_funnel.columns:
        st.subheader("Purchases by Category")
        cat_counts = df_funnel[df_funnel['EventType'] == "Purchase"]['Category'].value_counts()
        cat_fig = px.bar(cat_counts, x=cat_counts.index, y=cat_counts.values, title="Category Breakdown")
        st.plotly_chart(cat_fig, use_container_width=True)

st.caption("📊 Built with Streamlit & Plotly — Interactive Funnel Analytics")
