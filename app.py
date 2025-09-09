import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="E-commerce Funnel Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# --- Load Data ---
# Caches the data to avoid reloading on every interaction, improving performance.
@st.cache_data
def load_data():
    # Load the dataset from the CSV file
    df = pd.read_csv('ecommerce_clickstream_transactions.csv')
    return df

st.title("Loading Data...")
df = load_data()
st.title("E-commerce Funnel Dashboard 🛍️")


# --- Preprocessing ---
# 1. Handle missing values (if any). For this dataset, we can safely drop them.
df.dropna(inplace=True)

# 2. Convert 'Timestamp' column to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 3. Standardize event names for the funnel
# We are interested in product_view, add_to_cart, and purchase
event_name_map = {
    'product_view': 'Product View',
    'add_to_cart': 'Add to Cart',
    'purchase': 'Purchase'
}
df['EventType'] = df['EventType'].map(event_name_map)

# 4. Filter for main funnel events only
main_funnel_events = ['Product View', 'Add to Cart', 'Purchase']
df_funnel = df[df['EventType'].isin(main_funnel_events)]


# --- Display a sample of the cleaned data ---
st.header("Cleaned Funnel Data Sample")
st.dataframe(df_funnel.head())