import pandas as pd
import streamlit as st
import plotly.express as px

# Dashboard title
st.set_page_config(
    page_title="Pokemon description embeddings",
    page_icon="static/favicon.png",
    layout="wide",
)
st.title("Pokemon description embedding")
st.markdown(
    "This is a small app trained on most of the pokemon descriptions (flying excluded due to so few primary flyers)"
)

df = pd.read_csv("pokemon_embed.csv")

# Split into multiple columns
col1, col2 = st.columns(2)

fig_1 = px.scatter(
    df, x="Embed_0", y="Embed_1", color="primary_type", title="Normal Model Embeddings"
)
col1.plotly_chart(fig_1)

fig_2 = px.scatter(
    df,
    x="Embed_FineTune_0",
    y="Embed_FineTune_1",
    color="primary_type",
    title="Fine-Tuned Embeddings",
)
col2.plotly_chart(fig_2)
