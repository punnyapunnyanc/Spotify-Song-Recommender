import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Load dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load and preprocess
st.title("Spotify Song Recommender")
uploaded_file = st.file_uploader("Upload your Spotify dataset CSV", type="csv")

if uploaded_file is not None:
    spotify_data = load_data(uploaded_file)

    # Features to use
    features = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
    ]

    if not all(f in spotify_data.columns for f in features):
        st.error("Some required features are missing in your dataset.")
    else:
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(spotify_data[features])
        train_data, test_data = train_test_split(data_normalized, test_size=0.2, random_state=42)

        # Train model
        model = NearestNeighbors(n_neighbors=5)
        model.fit(train_data)

        st.subheader("Input Track Features")
        input_values = {}
        for feature in features:
            min_val = float(spotify_data[feature].min())
            max_val = float(spotify_data[feature].max())
            input_values[feature] = st.slider(feature.capitalize(), min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)

        input_df = pd.DataFrame([input_values])
        input_normalized = scaler.transform(input_df)

        distances, indices = model.kneighbors(input_normalized)
        recommendations = spotify_data.iloc[indices[0]]

        st.subheader("Recommended Songs")
        if {'artist_name', 'genre'}.issubset(recommendations.columns):
            if 'track_name' in recommendations.columns:
                st.dataframe(recommendations[['track_name', 'artist_name', 'genre']])
            else:
                st.dataframe(recommendations[['artist_name', 'genre']])
        else:
            st.dataframe(recommendations.head())

        # Accuracy function
        def calculate_recommendation_accuracy(test_data, model, threshold=0.2):
            total = len(test_data)
            good_matches = 0
            distances, _ = model.kneighbors(test_data)
            for dist_list in distances:
                if any(dist < threshold for dist in dist_list[1:]):
                    good_matches += 1
            accuracy = (good_matches / total) * 100
            return accuracy

        if st.button("Evaluate Accuracy"):
            acc = calculate_recommendation_accuracy(test_data, model)
            st.success(f"Distance-based Match Accuracy: {acc:.2f}%")
# Spotify-Song-Recommender
