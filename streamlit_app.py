import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
import io

st.set_page_config(page_title="Site-to-Site URL Matcher", layout="wide")
st.title("🔗 Site A vs Site B Page Matching")

# --- OpenAI API Key ---
openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")

# --- Upload CSVs ---
st.markdown("### 📂 Upload CSVs for Site A and Site B")
file_a = st.file_uploader("Upload Site A CSV", type=["csv"], key="site_a")
file_b = st.file_uploader("Upload Site B CSV", type=["csv"], key="site_b")

@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file)
    df['Embeddings'] = df['Embeddings'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    df['Keywords_Clean'] = df['Keywords'].astype(str).apply(lambda x: ' '.join(x.splitlines()))
    return df

@st.cache_data(show_spinner=False)
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Error embedding: {text[:30]}... -> {e}")
        return [0.0] * 1536

@st.cache_data(show_spinner=False)
def embed_column(texts, label):
    embeddings = []
    for i, text in enumerate(texts):
        embeddings.append(get_embedding(text))
        time.sleep(1)  # rate limit safety
    return embeddings

if file_a and file_b and openai.api_key:
    df_a = load_csv(file_a)
    df_b = load_csv(file_b)

    st.success("✅ Files uploaded and parsed.")
    st.markdown("### 🔄 Generating embeddings for Site A")
    with st.spinner("Embedding Site A..."):
        df_a['H1_Embedding'] = embed_column(df_a['H1'].astype(str).tolist(), "Site A H1")
        df_a['URL_Embedding'] = embed_column(df_a['URL'].astype(str).tolist(), "Site A URL")
        df_a['KW_Embedding'] = embed_column(df_a['Keywords_Clean'].tolist(), "Site A Keywords")

    st.markdown("### 🔄 Generating embeddings for Site B")
    with st.spinner("Embedding Site B..."):
        df_b['H1_Embedding'] = embed_column(df_b['H1'].astype(str).tolist(), "Site B H1")
        df_b['URL_Embedding'] = embed_column(df_b['URL'].astype(str).tolist(), "Site B URL")
        df_b['KW_Embedding'] = embed_column(df_b['Keywords_Clean'].tolist(), "Site B Keywords")

    def combine_embeddings(row, w_content=0.6, w_h1=0.2, w_kw=0.15, w_url=0.05):
        return (
            w_content * np.array(row['Embeddings']) +
            w_h1 * np.array(row['H1_Embedding']) +
            w_kw * np.array(row['KW_Embedding']) +
            w_url * np.array(row['URL_Embedding'])
        )

    df_a['Combined'] = df_a.apply(combine_embeddings, axis=1)
    df_b['Combined'] = df_b.apply(combine_embeddings, axis=1)

    st.markdown("### 📊 Matching pages based on combined embeddings")
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
            'Site A Keywords': row_a['Keywords'],
            'Best Match Site B URL': row_b['URL'],
            'Site B H1': row_b['H1'],
            'Site B Keywords': row_b['Keywords'],
            'Cosine Similarity (0–1)': round(similarity_matrix[i][best_idx], 4)
        })

    result_df = pd.DataFrame(results)
    st.markdown("### ✅ Matching complete")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Match Results as CSV", csv, "matched_results.csv", "text/csv")
