# app.py
# E-commerce Funnel Analytics Dashboard 🛍️
# Turns raw clickstream into a professional, interactive funnel dashboard:
# - KPIs + conversion rates
# - Funnel visualization + drop-off analysis
# - Trends over time
# - Activity heatmap (day/hour)
# - Segment conversions (Device/Category if present)
# - Product insights (top products, per-product conversion if possible)
# - Downloadable summary tables

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
    page_title="E-commerce Funnel Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------#
# Helpers
# ---------------------------#
ESSENTIAL_COLS = ["Timestamp", "EventType"]

def _safe_div(num, den):
    return (num / den) if den else 0.0

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to standardize common column name variants to Timestamp / EventType / UserID / ProductID / Category / Device."""
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}

    # Timestamp
    for key in ["timestamp", "event_time", "time", "datetime", "ts"]:
        if key in cols:
            rename_map[cols[key]] = "Timestamp"
            break

    # EventType
    for key in ["eventtype", "event", "event_name", "eventname", "name"]:
        if key in cols:
            rename_map[cols[key]] = "EventType"
            break

    # Optional / helpful columns
    for candidates, target in [
        (["userid", "user_id", "user", "uid"], "UserID"),
        (["productid", "product_id", "sku", "item_id"], "ProductID"),
        (["category", "product_category", "cat"], "Category"),
        (["device", "platform", "os"], "Device")
    ]:
        for key in candidates:
            if key in cols:
                rename_map[cols[key]] = target
                break

    return df.rename(columns=rename_map)

def standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Map any event variants into canonical funnel stages."""
    if "EventType" not in df.columns:
        return df

    # Lowercase for mapping
    raw = df["EventType"].astype(str).str.strip().str.lower()

    # Common variants
    mapping = {
        "product_view": "Product View",
        "view": "Product View",
        "product view": "Product View",
        "browse": "Product View",

        "add_to_cart": "Add to Cart",
        "addtocart": "Add to Cart",
        "cart_add": "Add to Cart",
        "cart add": "Add to Cart",

        "purchase": "Purchase",
        "order_completed": "Purchase",
        "order complete": "Purchase",
        "checkout_complete": "Purchase",
        "success": "Purchase"
    }

    # If already nice-cased values exist, keep them
    nice = df["EventType"].astype(str)
    # Prefer mapping; if not found, preserve original
    df["EventType"] = raw.map(mapping).fillna(nice)
    return df

def compute_funnel_counts(df_funnel: pd.DataFrame) -> pd.Series:
    stages = ["Product View", "Add to Cart", "Purchase"]
    counts = df_funnel["EventType"].value_counts().reindex(stages).fillna(0).astype(int)
    return counts

def build_conversion_table(counts: pd.Series) -> pd.DataFrame:
    views = int(counts.get("Product View", 0))
    carts = int(counts.get("Add to Cart", 0))
    purchases = int(counts.get("Purchase", 0))

    conv_view_to_cart = _safe_div(carts, views)
    conv_cart_to_purchase = _safe_div(purchases, carts)
    conv_view_to_purchase = _safe_div(purchases, views)

    drop_view = 1 - conv_view_to_cart
    drop_cart = 1 - conv_cart_to_purchase

    table = pd.DataFrame({
        "Stage": ["Product View", "Add to Cart", "Purchase"],
        "Count": [views, carts, purchases],
        "Conv. from Previous": [np.nan, conv_view_to_cart, conv_cart_to_purchase],
        "Conv. from Initial": [1.0, conv_view_to_cart, conv_view_to_purchase],
        "Drop-off from Previous": [np.nan, drop_view, drop_cart]
    })
    return table

def segment_conversion(df_funnel: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Compute per-segment conversion metrics."""
    stages = ["Product View", "Add to Cart", "Purchase"]
    if segment_col not in df_funnel.columns:
        return pd.DataFrame()

    # Count events per segment+stage
    grp = df_funnel.groupby([segment_col, "EventType"]).size().unstack("EventType").reindex(columns=stages, fill_value=0)
    grp = grp.fillna(0).astype(int)
    grp = grp.rename(columns={
        "Product View": "Views",
        "Add to Cart": "Carts",
        "Purchase": "Purchases"
    })

    # Conversions
    grp["View→Cart %"] = (grp["Carts"] / grp["Views"].replace(0, np.nan) * 100).fillna(0).round(2)
    grp["Cart→Purchase %"] = (grp["Purchases"] / grp["Carts"].replace(0, np.nan) * 100).fillna(0).round(2)
    grp["View→Purchase %"] = (grp["Purchases"] / grp["Views"].replace(0, np.nan) * 100).fillna(0).round(2)
    grp = grp.reset_index().rename(columns={segment_col: "Segment"})
    return grp

def product_conversion(df_funnel: pd.DataFrame) -> pd.DataFrame:
    """Per-Product conversions (requires ProductID)."""
    if "ProductID" not in df_funnel.columns:
        return pd.DataFrame()

    stages = ["Product View", "Add to Cart", "Purchase"]
    grp = df_funnel.groupby(["ProductID", "EventType"]).size().unstack("EventType").reindex(columns=stages, fill_value=0)
    grp = grp.fillna(0).astype(int)
    grp = grp.rename(columns={
        "Product View": "Views",
        "Add to Cart": "Carts",
        "Purchase": "Purchases"
    })
    grp["View→Cart %"] = (grp["Carts"] / grp["Views"].replace(0, np.nan) * 100).fillna(0).round(2)
    grp["Cart→Purchase %"] = (grp["Purchases"] / grp["Carts"].replace(0, np.nan) * 100).fillna(0).round(2)
    grp["View→Purchase %"] = (grp["Purchases"] / grp["Views"].replace(0, np.nan) * 100).fillna(0).round(2)
    grp = grp.reset_index()
    return grp

# ---------------------------#
# Data Load + Prep
# ---------------------------#
@st.cache_data
def load_data(path: str = "ecommerce_clickstream_transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_columns(df)
    # Ensure essential columns exist
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "EventType"])
    df = standardize_events(df)
    # Keep only main funnel stages in the dataset for analysis tabs; Overview tab can still reflect this subset
    main_funnel = ["Product View", "Add to Cart", "Purchase"]
    df_funnel = df[df["EventType"].isin(main_funnel)].copy()
    return df, df_funnel

try:
    df_raw, df_funnel_base = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# ---------------------------#
# Sidebar Filters
# ---------------------------#
st.sidebar.title("🔎 Filters")

# Date Range
min_d = df_funnel_base["Timestamp"].min().date()
max_d = df_funnel_base["Timestamp"].max().date()
date_from, date_to = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d, max_value=max_d
) if min_d and max_d else (date.today(), date.today())

mask_date = (df_funnel_base["Timestamp"].dt.date >= date_from) & (df_funnel_base["Timestamp"].dt.date <= date_to)
df_funnel = df_funnel_base.loc[mask_date].copy()

# Optional filters
if "Device" in df_funnel.columns:
    dev_sel = st.sidebar.multiselect("Device", sorted(df_funnel["Device"].dropna().unique().tolist()),
                                     default=sorted(df_funnel["Device"].dropna().unique().tolist()))
    if dev_sel:
        df_funnel = df_funnel[df_funnel["Device"].isin(dev_sel)]

if "Category" in df_funnel.columns:
    cat_sel = st.sidebar.multiselect("Category", sorted(df_funnel["Category"].dropna().unique().tolist()))
    if cat_sel:
        df_funnel = df_funnel[df_funnel["Category"].isin(cat_sel)]

if "ProductID" in df_funnel.columns:
    # Provide a quick search (optional)
    prod_search = st.sidebar.text_input("Search ProductID (contains)")
    if prod_search:
        df_funnel = df_funnel[df_funnel["ProductID"].astype(str).str.contains(prod_search, case=False, na=False)]

st.sidebar.markdown("---")
show_table_downloads = st.sidebar.checkbox("Enable downloads for summary tables", value=True)

# ---------------------------#
# Title + KPIs
# ---------------------------#
st.title("E-commerce Funnel Dashboard 🛍️")
st.caption("Analyze user behavior across the funnel to find drop-off points and improve conversion rates.")

funnel_counts = compute_funnel_counts(df_funnel)
views = int(funnel_counts.get("Product View", 0))
carts = int(funnel_counts.get("Add to Cart", 0))
purchases = int(funnel_counts.get("Purchase", 0))

conv_view_to_cart = _safe_div(carts, views)
conv_cart_to_purchase = _safe_div(purchases, carts)
conv_view_to_purchase = _safe_div(purchases, views)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Product Views", f"{views:,}")
col2.metric("Added to Cart", f"{carts:,}", f"{(conv_view_to_cart*100):.1f}% from views" if views else "—")
col3.metric("Purchases", f"{purchases:,}", f"{(conv_cart_to_purchase*100):.1f}% from carts" if carts else "—")
col4.metric("Overall Conversion", f"{(conv_view_to_purchase*100):.1f}%" if views else "0.0%")

# ---------------------------#
# Tabs
# ---------------------------#
tab_overview, tab_funnel, tab_trends, tab_segments, tab_products, tab_diagnostics = st.tabs(
    ["📊 Overview", "🔽 Funnel", "📈 Trends", "🧩 Segments", "🛒 Products", "🧪 Diagnostics"]
)

# ---------------------------#
# Overview Tab
# ---------------------------#
with tab_overview:
    left, right = st.columns((1, 1))
    with left:
        st.subheader("Event Distribution")
        if len(df_funnel) > 0:
            fig_pie = px.pie(
                df_funnel,
                names="EventType",
                title="Share of Funnel Events",
                hole=0.35
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data for the selected filters.")

    with right:
        st.subheader("Drop-off Insights")
        drop1 = (1 - conv_view_to_cart) * 100 if views else 0
        drop2 = (1 - conv_cart_to_purchase) * 100 if carts else 0
        st.markdown(
            f"""
            - 🔻 **{drop1:.1f}%** dropped after viewing (did not add to cart)  
            - 🔻 **{drop2:.1f}%** dropped after adding to cart (did not purchase)  
            - ✅ **{(conv_view_to_purchase*100):.1f}%** overall conversion from view → purchase
            """
        )

    st.subheader("Funnel Summary Table")
    conv_table = build_conversion_table(funnel_counts)
    st.dataframe(conv_table.style.format({
        "Conv. from Previous": "{:.2%}",
        "Conv. from Initial": "{:.2%}",
        "Drop-off from Previous": "{:.2%}"
    }), use_container_width=True)

    if show_table_downloads:
        st.download_button("Download Funnel Summary CSV", conv_table.to_csv(index=False).encode("utf-8"),
                           file_name="funnel_summary.csv", mime="text/csv")

# ---------------------------#
# Funnel Tab
# ---------------------------#
with tab_funnel:
    st.subheader("Funnel Visualization")
    stages = ["Product View", "Add to Cart", "Purchase"]
    fig_funnel = go.Figure(go.Funnel(
        y=stages,
        x=[views, carts, purchases],
        textposition="inside",
        textinfo="value+percent previous+percent initial"
    ))
    st.plotly_chart(fig_funnel, use_container_width=True)

# ---------------------------#
# Trends Tab
# ---------------------------#
with tab_trends:
    st.subheader("Daily Trends by Event")
    if len(df_funnel) > 0:
        daily = df_funnel.groupby([pd.Grouper(key="Timestamp", freq="D"), "EventType"]).size().reset_index(name="Count")
        fig_line = px.line(daily, x="Timestamp", y="Count", color="EventType", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

    st.subheader("Activity Heatmap (Day × Hour)")
    if len(df_funnel) > 0:
        tmp = df_funnel.copy()
        tmp["Day"] = tmp["Timestamp"].dt.day_name()
        tmp["Hour"] = tmp["Timestamp"].dt.hour
        # Order days Mon..Sun
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        tmp["Day"] = pd.Categorical(tmp["Day"], categories=day_order, ordered=True)
        heat = tmp.groupby(["Day", "Hour"]).size().reset_index(name="Count")
        fig_heat = px.density_heatmap(
            heat, x="Hour", y="Day", z="Count",
            nbinsx=24, histfunc="avg", title="Events by Day and Hour"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

# ---------------------------#
# Segments Tab
# ---------------------------#
with tab_segments:
    st.subheader("Conversion by Segment")

    segment_opts = []
    for c in ["Device", "Category"]:
        if c in df_funnel.columns:
            segment_opts.append(c)

    if len(segment_opts) == 0:
        st.info("No segment columns (Device/Category) found in the dataset.")
    else:
        seg_col = st.selectbox("Choose a segment column", segment_opts, index=0)
        seg_df = segment_conversion(df_funnel, seg_col)
        if seg_df.empty:
            st.info("Not enough data to compute segment conversions.")
        else:
            st.dataframe(seg_df, use_container_width=True)
            if show_table_downloads:
                st.download_button(f"Download {seg_col} Conversion CSV", seg_df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{seg_col.lower()}_conversion.csv", mime="text/csv")

            # Visualize View→Purchase % by segment (top 15)
            top = seg_df.sort_values("View→Purchase %", ascending=False).head(15)
            fig_bar = px.bar(top, x="Segment", y="View→Purchase %", title=f"{seg_col}: View→Purchase Conversion (%)")
            fig_bar.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------#
# Products Tab
# ---------------------------#
with tab_products:
    if "ProductID" not in df_funnel.columns:
        st.info("No ProductID column found for product-level insights.")
    else:
        st.subheader("Top Purchased Products")
        top_purchases = (
            df_funnel[df_funnel["EventType"] == "Purchase"]["ProductID"]
            .value_counts().head(20)
        )
        if len(top_purchases) > 0:
            fig_top = px.bar(
                x=top_purchases.index.astype(str),
                y=top_purchases.values,
                labels={"x": "ProductID", "y": "Purchases"},
                title="Top 20 Products by Purchases"
            )
            fig_top.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_top, use_container_width=True)

        st.subheader("Per-Product Conversion")
        prod_conv = product_conversion(df_funnel)
        if not prod_conv.empty:
            # Optional filter to focus on products with enough traffic
            min_views = st.slider("Minimum Views (to include in conversion table)", 0, int(prod_conv["Views"].max() or 0), 10)
            prod_conv_filtered = prod_conv[prod_conv["Views"] >= min_views].copy()
            st.dataframe(prod_conv_filtered.sort_values("View→Purchase %", ascending=False), use_container_width=True)

            if show_table_downloads:
                st.download_button("Download Per-Product Conversion CSV",
                                   prod_conv_filtered.to_csv(index=False).encode("utf-8"),
                                   file_name="product_conversion.csv", mime="text/csv")
        else:
            st.info("Not enough product-level data to compute conversions.")

# ---------------------------#
# Diagnostics Tab
# ---------------------------#
with tab_diagnostics:
    st.subheader("Cleaned Funnel Data Sample")
    st.dataframe(df_funnel.head(50), use_container_width=True)

    st.subheader("Column Summary")
    st.write(pd.DataFrame({
        "Column": df_raw.columns,
        "Non-Null Count": [df_raw[c].notna().sum() for c in df_raw.columns],
        "Dtype": [df_raw[c].dtype for c in df_raw.columns]
    }))

    st.subheader("Raw Event Counts (All Events)")
    st.write(df_raw["EventType"].value_counts())

    st.caption("Diagnostics are for quick verification and debugging of the pipeline.")
