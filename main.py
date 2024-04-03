import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# UI Setup
st.title('Semantic Similarity of Clusters')
st.markdown('This app analyzes clusters for semantic similarity.')

uploaded_file = st.file_uploader("Upload your CSV file (cluster_name.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()  

threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

# Model Setup
modelName = "BAAI/bge-m3"
st.markdown("**Model:** {}".format(modelName))

# Button Functionality
if st.button("Analyze Clusters"):
    st.subheader("Loading Model...")
    model = SentenceTransformer(modelName, trust_remote_code=True)
    st.success("Model Loaded!")

    # Similarity calculations 
    st.subheader("Calculating Similarities...") 
    cluster_embeddings = model.encode(df['cluster'].to_list(), convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(cluster_embeddings, cluster_embeddings)
    above_threshold_indices = np.where(np.triu(similarity_matrix.numpy(), k=1) > threshold)

    related_pairs = []
    for i, j in zip(*above_threshold_indices):
        cluster1 = df['cluster'].iloc[i]
        cluster2 = df['cluster'].iloc[j]
        similarity_score = similarity_matrix[i, j].item()
        related_pairs.append((cluster1, cluster2, similarity_score))

    links_df = pd.DataFrame(related_pairs, columns=['Cluster_1', 'Cluster_2', f'Similarity_{modelName}'])


    # Download Button
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(links_df)
    st.download_button(
        label="Download Similarity Data as CSV",
        data=csv,
        file_name='cluster_similarity_data.csv',
        mime='text/csv',
    ) 
