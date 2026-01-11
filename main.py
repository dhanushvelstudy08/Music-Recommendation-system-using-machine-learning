import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
from sklearn.preprocessing import LabelEncoder

# Set up Spotify API credentials (replace with your actual credentials)
client_id = ''  # Replace with your Spotify Client ID
client_secret = ''  # Replace with your Spotify Client Secret

# Spotify Client setup
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Read the model metrics from the file
with open('model_metrics.txt', 'r') as f:
    metrics = f.read()

# Updated fetch_song_data function
def fetch_song_data(song_name):
    results = sp.search(q=song_name, limit=50)  # Limit to 50 songs for now
    songs = []

    if not results['tracks']['items']:
        print(f"No results found for '{song_name}'.")
        return []

    for track in results['tracks']['items']:
        song_details = {
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'popularity': track['popularity'],
            'genres': track['artists'][0].get('genres', ['Unknown'])[0],
            'id': track['id'],
            'url': track['external_urls']['spotify'],  # Spotify link to the song
            'image': track['album']['images'][0]['url'] if track['album']['images'] else ''  # Album image
        }
        songs.append(song_details)

    return songs

# Function to preprocess the song data
def preprocess_data(df):
    df['genres'] = df['genres'].fillna('Unknown')
    le = LabelEncoder()
    df['genres_encoded'] = le.fit_transform(df['genres'])
    df['popularity_scaled'] = df['popularity'] / df['popularity'].max()  # Normalize popularity
    
    def independent_func(popularity):
        return 1 if popularity < 20 else 0  # Independent songs with low popularity
    
    df['is_independent'] = df['popularity'].apply(independent_func)  # Apply the function
    
    # Features for the model
    X = df[['genres_encoded', 'popularity_scaled', 'is_independent']]
    y = df['is_independent']  # Target variable: binary classification
    return X, y

# Streamlit UI
st.title("Music Recommendation System")

# Display the model metrics
st.write(f"Model Metrics:\n{metrics}")

# Text input to search for songs
song_name = st.text_input("Enter Song Name", "")

if song_name:
    # Fetch song data
    song_data_list = fetch_song_data(song_name)

    if not song_data_list:
        st.write("No results found for your search.")
    else:
        # Display song details in the Streamlit app
        song_df = pd.DataFrame(song_data_list)
        X, y = preprocess_data(song_df)

        # Predict independent or popular for each song
        predictions = model.predict(X)

        # Add predictions to the song data for displaying
        song_df['prediction'] = predictions

        # Sort the DataFrame by popularity in ascending order
        song_df = song_df.sort_values(by='popularity', ascending=True)

        # Create columns to display the songs in a grid (5 per row)
        for i in range(0, len(song_df), 5):
            cols = st.columns(5)  # 5 columns per row
            for j, col in enumerate(cols):
                index = i + j
                if index < len(song_df):
                    row = song_df.iloc[index]
                    
                    # Optional: Display the album image if you want to add it
                    col.image(row['image'], caption=f"Album: {row['name']}", use_container_width=True)
                    # Display song name and artist as clickable text
                    col.markdown(f"[**{row['name']}** by *{row['artist']}*]({row['url']})")
                    col.write("---")  # Add separator between song details
else:
    st.write("Search for a song to get recommendations.")

