import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from io import StringIO

st.set_page_config(page_title="URL Matcher", layout="wide")
st.title("🔗 Site A vs Site B URL Matching")

# --- Step 1: User enters their own OpenAI key ---
openai_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
if not openai_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()
openai.api_key = openai_key

# --- Step 2: Upload CSVs ---
st.markdown("### 📂 Upload CSVs for Site A and Site B")
file_a = st.file_uploader("Upload Site A CSV", type="csv", key="site_a")
file_b = st.file_uploader("Upload Site B CSV", type="csv", key="site_b")

# --- Helper functions ---
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = openai.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error embedding: {text[:30]}... → {e}")
        return [0.0] * 1536

def embed_column(texts, label):
    embeddings = []
    with st.spinner(f"Embedding {label}..."):
        for i, text in enumerate(texts):
            embeddings.append(get_embedding(str(text)))
            time.sleep(1)  # rate limit buffer
    return embeddings

def combine_embeddings(row, w_content=0.6, w_h1=0.2, w_kw=0.15, w_url=0.05):
    return (
        w_content * np.array(row['Embeddings']) +
        w_h1 * np.array(row['H1_Embedding']) +
        w_kw * np.array(row['KW_Embedding']) +
        w_url * np.array(row['URL_Embedding'])
    )

# --- Step 3: When files are uploaded ---
if file_a and file_b:
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    # Parse embeddings
    df_a['Embeddings'] = df_a['Embeddings'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    df_b['Embeddings'] = df_b['Embeddings'].apply(lambda x: np.array([float(i) for i in x.split(',')]))

    # Clean keyword columns
    df_a['Keywords_Clean'] = df_a['Keywords'].astype(str).apply(lambda x: ' '.join(x.splitlines()))
    df_b['Keywords_Clean'] = df_b['Keywords'].astype(str).apply(lambda x: ' '.join(x.splitlines()))

    # Embed H1, URL, and Keywords
    st.subheader("🔄 Generating embeddings for Site A")
    df_a['H1_Embedding'] = embed_column(df_a['H1'], "Site A H1")
    df_a['URL_Embedding'] = embed_column(df_a['URL'], "Site A URL")
    df_a['KW_Embedding'] = embed_column(df_a['Keywords_Clean'], "Site A Keywords")

    st.subheader("🔄 Generating embeddings for Site B")
    df_b['H1_Embedding'] = embed_column(df_b['H1'], "Site B H1")
    df_b['URL_Embedding'] = embed_column(df_b['URL'], "Site B URL")
    df_b['KW_Embedding'] = embed_column(df_b['Keywords_Clean'], "Site B Keywords")

    # Combine and compute similarity
    st.subheader("🔁 Calculating Matches")
    df_a['Combined'] = df_a.apply(combine_embeddings, axis=1)
    df_b['Combined'] = df_b.apply(combine_embeddings, axis=1)

    emb_a = np.stack(df_a['Combined'].values)
    emb_b = np.stack(df_b['Combined'].values)
    similarity_matrix = cosine_similarity(emb_a, emb_b)

    # Generate best match for each Site A URL
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

    # Output results
    st.subheader("✅ Match Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "matched_results.csv", "text/csv")
