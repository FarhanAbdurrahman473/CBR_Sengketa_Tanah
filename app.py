import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(
    page_title="CBR Sengketa Tanah",
    page_icon="⚖️",
    layout="wide"
)

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed/cases.csv")


@st.cache_resource
def load_vectorizer():
    return joblib.load("models/tfidf_vectorizer.pkl")


df = load_dataset()
vectorizer = load_vectorizer()

X = vectorizer.transform(df["ringkasan_fakta"])

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Tentang Sistem")

st.sidebar.info(
    """
Sistem ini menggunakan metode:

- Case-Based Reasoning (CBR)
- TF-IDF
- Cosine Similarity

Dataset:
40 Putusan Sengketa Tanah
"""
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Jumlah Kasus",
    len(df)
)

st.sidebar.metric(
    "Jumlah Label",
    df["label"].nunique()
)

# ======================================================
# RETRIEVAL
# ======================================================

def retrieve_cases(query, top_k=5):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        X
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    results["similarity"] = similarities[top_indices]

    return results


# ======================================================
# REUSE
# ======================================================

def predict_case(query):

    retrieved = retrieve_cases(query)

    prediction = Counter(
        retrieved["label"]
    ).most_common(1)[0][0]

    return prediction, retrieved


# ======================================================
# HEADER
# ======================================================

st.title("⚖️ Sistem Case-Based Reasoning")

st.subheader(
    "Prediksi Putusan Sengketa Tanah"
)

st.markdown(
"""
Masukkan kronologi singkat suatu sengketa tanah,
kemudian sistem akan mencari kasus yang paling mirip
dan memberikan prediksi putusan.
"""
)

# ======================================================
# INPUT
# ======================================================

query = st.text_area(
    "Masukkan Kronologi Sengketa",
    height=250,
    placeholder="""
Contoh:

Penggugat membeli sebidang tanah dari tergugat,
namun sertifikat hak milik belum dibalik nama
karena tergugat tidak diketahui keberadaannya.
"""
)

# ======================================================
# BUTTON
# ======================================================

if st.button("🔍 Prediksi Putusan"):

    if query.strip() == "":

        st.warning("Silakan masukkan kronologi terlebih dahulu.")

    else:

        with st.spinner("Mencari kasus paling mirip..."):

            prediction, retrieved = predict_case(query)

        st.success(
            f"Prediksi Putusan : **{prediction}**"
        )

        st.markdown("---")

        st.subheader("Top 5 Kasus Paling Mirip")

        for i, row in retrieved.iterrows():

            with st.expander(
                f"{row['nomor_perkara']} | Similarity {row['similarity']:.2%}"
            ):

                st.write(
                    f"**Label Putusan :** {row['label']}"
                )

                st.write(
                    f"**Penggugat :** {row['penggugat']}"
                )

                st.write(
                    f"**Tergugat :** {row['tergugat']}"
                )

                st.write(
                    f"**Similarity :** {row['similarity']:.4f}"
                )

                st.markdown("### Ringkasan Fakta")

                st.write(
                    row["ringkasan_fakta"]
                )

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.caption(
    "Case-Based Reasoning untuk Prediksi Putusan Sengketa Tanah menggunakan TF-IDF dan Cosine Similarity."
)