import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Spotify API credentials (replace with your actual credentials)
client_id = 'b152015de6df4416ab6723acdf1eef6a'  # Replace with your Spotify Client ID
client_secret = 'ec045482f65b47f08766bc9e8db43015'  # Replace with your Spotify Client Secret

# Spotify Client setup
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to fetch song data from Spotify API
def fetch_song_data(song_name):
    results = sp.search(q=song_name, limit=50)  # Limit to 50 songs for now
    songs = []

    for track in results['tracks']['items']:
        song_details = {
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'popularity': track['popularity'],
            'genres': track['artists'][0].get('genres', ['Unknown'])[0],  # Assuming first genre
            'id': track['id']
        }
        songs.append(song_details)

    return songs

# Function to preprocess the song data
def preprocess_data(df):
    # Fill missing values and encode categorical variables
    df['genres'] = df['genres'].fillna('Unknown')
    le = LabelEncoder()
    df['genres_encoded'] = le.fit_transform(df['genres'])
    df['popularity_scaled'] = df['popularity'] / df['popularity'].max()  # Normalize popularity
    
    # Create a binary feature for independent songs based on popularity threshold
    df['is_independent'] = df['popularity'].apply(lambda x: 1 if x < 20 else 0)  # Independent songs with low popularity
    
    # Features for the model
    X = df[['genres_encoded', 'popularity_scaled', 'is_independent']]
    return X

# Fetch data for a set of songs (just an example of how to prepare your dataset)
song_names = ['Shape of You', 'Blinding Lights', 'Billie Jean', 'Smells Like Teen Spirit']
song_data_list = []

for song_name in song_names:
    song_data_list += fetch_song_data(song_name)

# Convert song data to DataFrame
song_df = pd.DataFrame(song_data_list)

# Preprocess the data
X = preprocess_data(song_df)

# Let's set the target variable 'y' as the popularity of the songs (for prediction purpose)
y = song_df['popularity']  # This could be adjusted depending on what you're predicting

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier with class balancing
rf = RandomForestClassifier(random_state=100, class_weight='balanced')

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Save the model
joblib.dump(best_rf, 'random_forest_model.pkl')

# Make predictions on the test set
y_pred = best_rf.predict(X_test)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f'Model Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

with open('model_metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}\n')
    f.write(f'Precision: {precision:.2f}\n')
    f.write(f'Recall: {recall:.2f}\n')
    f.write(f'F1 Score: {f1:.2f}\n')
