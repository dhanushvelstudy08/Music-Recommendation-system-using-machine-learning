'''import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
from requests.exceptions import ReadTimeout

# ✅ Spotify API Credentials
CLIENT_ID = 'b152015de6df4416ab6723acdf1eef6a'
CLIENT_SECRET = 'ec045482f65b47f08766bc9e8db43015'

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# ✅ Load Dataset
df = pd.read_csv('music_data.csv')

# ✅ Sample Data (Reduced from 10,000 → 5,000)
df = df.sample(n=min(5000, len(df)), random_state=42)  # Prevent excessive memory usage

# ✅ Handle Missing Values
df.fillna(0, inplace=True)

# ✅ Process Playlist Names
if 'playlists' in df.columns:
    # Extract playlist names if available
    df['playlist_names'] = df['playlists'].apply(lambda x: x[0]['name'] if isinstance(x, list) and x and 'name' in x[0] else 'Unknown')
    label_encoder = LabelEncoder()
    df['playlist_encoded'] = label_encoder.fit_transform(df['playlist_names'])
else:
    print("Error: 'playlists' column not found.")

# ✅ Spotify Data Fetch Function (with caching)
spotify_cache = {}

def fetch_spotify_data(track_name, retries=3, delay=5):
    if track_name in spotify_cache:
        return spotify_cache[track_name]
    
    for attempt in range(retries):
        try:
            results = sp.search(q=track_name, type='track', limit=1)
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_id = track['id']
                track_popularity = track['popularity']
                spotify_cache[track_name] = (track_id, track_popularity)
                return track_id, track_popularity
            return None, None
        except ReadTimeout:
            print(f"ReadTimeout occurred. Retrying {attempt + 1}/{retries}...")
            time.sleep(delay)
    
    return None, None

# ✅ Apply Spotify API Calls (to get spotify_popularity and track_id)
df['spotify_track_id'], df['spotify_popularity'] = zip(*df['name'].apply(fetch_spotify_data))

# ✅ Set Independent Artist Criteria (Assign 1 if popularity < 50, else 0)
df['is_independent'] = df['spotify_popularity'].apply(lambda x: 1 if x < 50 else 0)

# ✅ Define Features & Target
X = df.drop(columns=['date', 'version', 'playlists', 'name', 'description', 'playlist_names', 'spotify_track_id', 'spotify_popularity'], errors='ignore')
y = df['is_independent']

# ✅ Handle Non-Numeric Features (select only numeric features)
X = X.select_dtypes(include=[np.number])  # Drop non-numeric columns

# ✅ Balance Dataset Using SMOTE (if needed)
from imblearn.over_sampling import SMOTE

if y.value_counts().min() / y.value_counts().max() < 0.5:  # Only balance if there's imbalance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

# ✅ Split Dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=65, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Save Trained Model
import pickle
with open('music_recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# ✅ Predict & Evaluate Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Hyperparameter Tuning (optional)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# ✅ Print Best Parameters
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_}")

# ✅ Save Best Model
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(grid_search.best_estimator_, model_file)

print("🎉 Model training completed and saved.")

# ✅ Feature Importance Visualization
import matplotlib.pyplot as plt
feature_importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Model")
plt.show()'''
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
