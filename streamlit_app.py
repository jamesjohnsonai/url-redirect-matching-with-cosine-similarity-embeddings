import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

st.set_page_config(page_title="AI Redirect Mapper", layout="wide")
st.title("🤖 AI-Powered Semantic Redirect Mapper")

with st.expander("ℹ️ About this tool"):
    st.markdown(""" 
    This tool matches URLs from two different sites using **semantic similarity** — not just string comparison.

    Powered by OpenAI embeddings, it compares pages based on:
    - Full page content
    - H1 tags
    - Top 5 organic keywords
    - URL context

    It's ideal for:
    - Website migrations
    - Domain consolidations
    - Semantic SEO audits
    - Redirect mapping that goes beyond fuzzy matching

    🔐 You provide your own OpenAI API key. Nothing is stored.
    """)

with st.expander("📋 How to use"):
    st.markdown("""
    1. Crawl **both sites** using [Screaming Frog](https://www.screamingfrog.co.uk/seo-spider/)
        - Make sure to extract: `URL`, `H1`, OpenAI `Embeddings` (via JavaScript integration), and `Keywords`

    2. Format both CSVs like this:
    ```
    | URL | H1 | Embeddings | Keywords |
    ```
    - `Embeddings`: 1536-dim OpenAI vectors in a single comma-separated string
    - `Keywords`: newline-separated (each on a new line within the same cell)

    3. Upload the two CSVs — one for **Site A** and one for **Site B**
    4. Enter your **OpenAI API key** (kept private)
    5. Click "Download CSV" after results are generated
    """)

with st.expander("❓ Why not just use fuzzy matching?"):
    st.markdown("""
    Traditional redirect tools rely on exact or fuzzy string matching. That’s fast, but shallow.

    This tool uses **OpenAI embeddings** to understand the meaning behind the content.
    - It can match "Pest control near me" ↔ "Exterminator services"
    - Works great for different phrasings across domains

    🔍 Think of it as redirect mapping powered by language understanding.
    """)

# --- Step 1: User enters their own OpenAI key ---
api_key_input = st.text_input("🔑 Enter your OpenAI API key", type="password")
if api_key_input:
    st.session_state.api_key = api_key_input

if "api_key" not in st.session_state or not st.session_state.api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

openai.api_key = st.session_state.api_key

# --- Step 2: Upload CSVs ---
st.markdown("### 📂 Upload CSVs")
file_a = st.file_uploader("Upload Site A CSV", type="csv", key="site_a")
file_b = st.file_uploader("Upload Site B CSV", type="csv", key="site_b")

# --- Helper functions ---
def batch_get_embeddings(text_list, label):
    key = f"{label}_embeddings"
    if key in st.session_state:
        use_cached = st.checkbox(f"Use cached embeddings for {label}?", value=True)
        if use_cached:
            return st.session_state[key]

    batch_size = 50
    results = []
    progress = st.progress(0)
    total = len(text_list)
    for i in range(0, total, batch_size):
        batch = text_list[i:i + batch_size]
        try:
            response = openai.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings = [item.embedding for item in response.data]
        except Exception as e:
            st.error(f"Batch failed ({label}): {e}")
            embeddings = [[0.0]*1536 for _ in batch]
        results.extend(embeddings)
        progress.progress(min((i + batch_size) / total, 1.0))
        time.sleep(1)

    st.session_state[key] = results
    return results

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

    df_a['Embeddings'] = df_a['Embeddings'].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    df_b['Embeddings'] = df_b['Embeddings'].apply(lambda x: np.array([float(i) for i in x.split(',')]))

    df_a['Keywords_Clean'] = df_a['Keywords'].astype(str).apply(lambda x: ' '.join(x.splitlines()))
    df_b['Keywords_Clean'] = df_b['Keywords'].astype(str).apply(lambda x: ' '.join(x.splitlines()))

    st.subheader("🔄 Generating embeddings for Site A")
    df_a['H1_Embedding'] = batch_get_embeddings(df_a['H1'].astype(str).tolist(), "Site A H1")
    df_a['URL_Embedding'] = batch_get_embeddings(df_a['URL'].astype(str).tolist(), "Site A URL")
    df_a['KW_Embedding'] = batch_get_embeddings(df_a['Keywords_Clean'].tolist(), "Site A Keywords")

    st.subheader("🔄 Generating embeddings for Site B")
    df_b['H1_Embedding'] = batch_get_embeddings(df_b['H1'].astype(str).tolist(), "Site B H1")
    df_b['URL_Embedding'] = batch_get_embeddings(df_b['URL'].astype(str).tolist(), "Site B URL")
    df_b['KW_Embedding'] = batch_get_embeddings(df_b['Keywords_Clean'].tolist(), "Site B Keywords")

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
            'Site A Keywords': row_a['Keywords'],
            'Best Match Site B URL': row_b['URL'],
            'Site B H1': row_b['H1'],
            'Site B Keywords': row_b['Keywords'],
            'Cosine Similarity (0–1)': round(similarity_matrix[i][best_idx], 4)
        })

    st.subheader("✅ Match Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "matched_results.csv", "text/csv")