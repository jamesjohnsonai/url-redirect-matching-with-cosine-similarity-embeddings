import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="AI Redirect Mapper", layout="centered")
st.markdown("""
<style>
    .step-box {
        background-color: #f0f2f6;
        padding: 1.25rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 6px solid #4a90e2;
    }
    .step-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #4a90e2;
    }
    .step-text {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered Semantic Redirect Mapper")

with st.expander("ℹ️ About this tool"):
    st.markdown(""" 
    This tool matches URLs from two different sites using **semantic similarity** — not just string comparison.

    Powered by OpenAI embeddings, it compares pages based on:
    - Full page content
    - H1 tags
    - URL context (path, slug)

    It's ideal for:
    - Website migrations
    - Domain consolidations
    - Semantic SEO audits
    - Redirect mapping that goes beyond fuzzy matching

    🧠 It does **not rely on exact keyword or string matching**, but instead uses language models to understand meaning.

    🔐 You provide your own OpenAI API key. Nothing is stored.
    """)

st.markdown("""
<div class='step-box'>
    <div class='step-title'>Step 1: Enter Your OpenAI API Key</div>
    <div class='step-text'>
        Paste your OpenAI key below to authenticate embedding generation. Your key is never stored.
    </div>
</div>
""", unsafe_allow_html=True)

api_key_input = st.text_input("🔑 OpenAI API key", type="password")
if api_key_input:
    st.session_state.api_key = api_key_input

if "api_key" not in st.session_state or not st.session_state.api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

openai.api_key = st.session_state.api_key

st.markdown("""
<div class='step-box'>
    <div class='step-title'>Step 2: Upload Your Site CSV Files</div>
    <div class='step-text'>
        Upload one file for Site A and one for Site B.
    </div>
</div>
""", unsafe_allow_html=True)

file_a = st.file_uploader("📄 Upload Site A CSV", type="csv", key="site_a")
file_b = st.file_uploader("📄 Upload Site B CSV", type="csv", key="site_b")

if file_a and file_b:
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    st.markdown("### 🧭 Map Site A Columns")
    col_url_a = st.selectbox("Site A: Select the URL column", df_a.columns, key="col_url_a")
    col_h1_a = st.selectbox("Site A: Select the H1 column", df_a.columns, key="col_h1_a")
    col_emb_a = st.selectbox("Site A: Select the Embeddings column", df_a.columns, key="col_emb_a")

    st.markdown("### 🧭 Map Site B Columns")
    col_url_b = st.selectbox("Site B: Select the URL column", df_b.columns, key="col_url_b")
    col_h1_b = st.selectbox("Site B: Select the H1 column", df_b.columns, key="col_h1_b")
    col_emb_b = st.selectbox("Site B: Select the Embeddings column", df_b.columns, key="col_emb_b")

    if st.button("✅ Confirm Column Mapping"):
        st.session_state.upload_ready = True
        st.session_state.df_a = df_a
        st.session_state.df_b = df_b
        st.session_state.col_url_a = col_url_a
        st.session_state.col_h1_a = col_h1_a
        st.session_state.col_emb_a = col_emb_a
        st.session_state.col_url_b = col_url_b
        st.session_state.col_h1_b = col_h1_b
        st.session_state.col_emb_b = col_emb_b

if 'upload_ready' in st.session_state and st.session_state.upload_ready:
    # Apply column mappings
    df_a['URL'] = df_a[col_url_a]
    df_a['H1'] = df_a[col_h1_a]
    df_a['Embeddings'] = df_a[col_emb_a].apply(safe_embedding_parse)

    df_b['URL'] = df_b[col_url_b]
    df_b['H1'] = df_b[col_h1_b]
    df_b['Embeddings'] = df_b[col_emb_b].apply(safe_embedding_parse)

    st.subheader("🔄 Generating embeddings for Site A")
    df_a['H1_Embedding'] = batch_get_embeddings(df_a['H1'].astype(str).tolist(), "Site A H1")
    df_a['URL_Embedding'] = batch_get_embeddings(df_a['URL'].astype(str).tolist(), "Site A URL")

    st.subheader("🔄 Generating embeddings for Site B")
    df_b['H1_Embedding'] = batch_get_embeddings(df_b['H1'].astype(str).tolist(), "Site B H1")
    df_b['URL_Embedding'] = batch_get_embeddings(df_b['URL'].astype(str).tolist(), "Site B URL")

    st.subheader("🔁 Calculating Matches")
    df_a['Combined'] = df_a.apply(combine_embeddings, axis=1)
    df_b['Combined'] = df_b.apply(combine_embeddings, axis=1)

    emb_a = np.stack(df_a['Combined'].values)
    emb_b = np.stack(df_b['Combined'].values)
    similarity_matrix = cosine_similarity(emb_a, emb_b)

    results = []
    for i, row_a in df_a.iterrows():
        best_idx = np.argmax(similarity_matrix[i])
        row_b = df_b.iloc[best_idx]
        results.append({
            'Site A URL': row_a['URL'],
            'Site A H1': row_a['H1'],
            'Best Match Site B URL': row_b['URL'],
            'Site B H1': row_b['H1'],
            'Cosine Similarity (0–1)': round(similarity_matrix[i][best_idx], 4)
        })

    st.subheader("✅ Match Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "matched_results.csv", "text/csv")
