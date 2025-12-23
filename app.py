import streamlit as st
import pandas as pd
from datetime import datetime
from transformers import pipeline

# ---------- Load sentiment analysis model (Hugging Face) ----------
@st.cache_resource
def load_sentiment_model():
    """
    Load the Hugging Face sentiment-analysis pipeline with the recommended model.
    This will raise an exception if the package or model cannot be loaded.
    """
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


sentiment_analyzer = load_sentiment_model()

# ---------- 1. Load data ----------

@st.cache_data
def load_data():
    products = pd.read_csv("products.csv")
    testimonials = pd.read_csv("testimonials.csv")
    reviews = pd.read_csv("reviews.csv")

    # Parse dates in reviews
    # Adjust column name if your scraped file uses something else
    if "date_parsed" in reviews.columns:
        reviews["date_parsed"] = pd.to_datetime(reviews["date_parsed"], errors="coerce")
    elif "date_raw" in reviews.columns:
        reviews["date_parsed"] = pd.to_datetime(reviews["date_raw"], errors="coerce")
    else:
        # Fallback: try to parse any column that looks like a date
        for col in reviews.columns:
            if "date" in col.lower():
                reviews["date_parsed"] = pd.to_datetime(reviews[col], errors="coerce")
                break

    return products, testimonials, reviews


products_df, testimonials_df, reviews_df = load_data()


# ---------- 2. Sidebar navigation ----------

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select section:",
    ("Products", "Testimonials", "Reviews")
)


st.title("Web Scraping Dev – Data Explorer")


# ---------- 3. Products section ----------

if section == "Products":
    st.header("Products")

    st.write("Below is the scraped products dataset:")
    st.dataframe(products_df)


# ---------- 4. Testimonials section ----------

elif section == "Testimonials":
    st.header("Testimonials")

    st.write("Below are the scraped testimonials:")
    st.dataframe(testimonials_df)


# ---------- 5. Reviews section (core feature) ----------

elif section == "Reviews":
    st.header("Reviews")

    if "date_parsed" not in reviews_df.columns:
        st.error("No parsed date column found in reviews data.")
    else:
        # Keep only rows with valid dates
        reviews_valid = reviews_df.dropna(subset=["date_parsed"]).copy()
        # Focus on 2023 only (use .copy() to avoid SettingWithCopyWarning)
        reviews_2023 = reviews_valid[reviews_valid["date_parsed"].dt.year == 2023].copy()
        # Filter to only include reviews up to May 2023
        reviews_2023 = reviews_2023[reviews_2023["date_parsed"].dt.month <= 5].copy()
        if reviews_2023.empty:
            st.warning("No reviews found for January to May 2023.")
        else:
            # Create a month label like "2023-01", "2023-02", ...
            reviews_2023["year_month"] = reviews_2023["date_parsed"].dt.to_period("M")
            # Build full list of months for 2023 (Jan-May) so slider always shows entire period
            month_options = pd.period_range(start="2023-01", end="2023-05", freq="M")
            # Turn Period objects into nice strings like "January 2023"
            month_display = [m.strftime("%B %Y") for m in month_options]
            month_map = dict(zip(month_display, month_options))

            # ---------- Month selection widget ----------
            selected_month_label = st.select_slider(
                "Select a month in 2023:",
                options=month_display
            )

            selected_period = month_map[selected_month_label]
            # Filter by chosen month
            filtered = reviews_2023[reviews_2023["year_month"] == selected_period]
            st.subheader(f"Reviews for {selected_month_label}")
            st.write(f"Number of reviews: {len(filtered)}")

            # Sentiment classification using Hugging Face transformers
            if "text" in filtered.columns:
                try:
                    texts = filtered["text"].astype(str).tolist()
                    preds = sentiment_analyzer(texts, truncation=True)
                    labels = [p["label"] for p in preds]
                    scores = [p.get("score") for p in preds]

                    mapped = ["Positive" if lbl.upper().startswith("POS") else "Negative" for lbl in labels]
                    filtered = filtered.copy()
                    filtered["sentiment"] = mapped
                    filtered["sentiment_score"] = scores

                    # Aggregate counts and average confidence per sentiment
                    agg = (
                        filtered.groupby("sentiment")["sentiment_score"]
                        .agg(count="size", avg_score="mean")
                        .reset_index()
                    )

                    # Ensure both Positive and Negative appear in agg
                    for s in ("Positive", "Negative"):
                        if s not in agg["sentiment"].values:
                            agg = pd.concat([agg, pd.DataFrame({"sentiment":[s], "count":[0], "avg_score":[0.0]})], ignore_index=True)

                    # Display metrics
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Reviews in Month", len(filtered))
                    pos_row = agg[agg["sentiment"]=="Positive"].iloc[0]
                    neg_row = agg[agg["sentiment"]=="Negative"].iloc[0]
                    c2.metric("Positive", int(pos_row["count"]))
                    c3.metric("Negative", int(neg_row["count"]))

                    # Build a modern-looking bar chart with labeled average confidence
                    selection = alt.selection_single(on="mouseover", fields=["sentiment"], empty="none")

                    base = alt.Chart(agg).encode(
                        x=alt.X("sentiment:N", title=None, sort=["Positive", "Negative"]),
                        y=alt.Y("count:Q", title="Count", axis=alt.Axis(grid=False)),
                        color=alt.Color(
                            "sentiment:N",
                            scale=alt.Scale(domain=["Positive", "Negative"], range=["#2ca02c", "#d62728"]),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("sentiment:N", title="Sentiment"),
                            alt.Tooltip("count:Q", title="Count"),
                            alt.Tooltip("avg_score:Q", title="Avg Confidence", format=".3f"),
                        ],
                    )

                    bars = base.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, opacity=0.9).add_selection(selection)
                    labels = base.mark_text(dy=-10, color="#111", fontSize=12).encode(
                        text=alt.Text("count:Q", format="d")
                    )

                    conf_labels = base.mark_text(dy=14, color="#222", fontSize=11).encode(
                        text=alt.Text("avg_score:Q", format=".1%")
                    ).transform_filter(selection)

                    chart = (bars + labels + conf_labels).properties(height=320).configure_view(stroke=None)
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Sentiment classification failed: {e}")
            else:
                st.info("No review text column found; skipping sentiment classification.")

            # Optional: show avg rating if you scraped stars
            if "stars" in filtered.columns:
                avg_rating = filtered["stars"].mean()
                st.write(f"Average rating: {avg_rating:.2f} / 5")

            # Show filtered reviews in a table
            st.dataframe(filtered)
