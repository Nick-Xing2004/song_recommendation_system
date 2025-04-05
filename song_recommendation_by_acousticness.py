import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#predict and recommend songs based on song's acousticness, instrumentalness, and speechiness

def train_model():  
    df_song=pd.read_csv('./data.csv')   #read in data
    X=df_song[['acousticness', 'instrumentalness', 'speechiness']].to_numpy()

    #standardize the features
    scaler=StandardScaler()
    X_standardized=scaler.fit_transform(X)
    return X_standardized, df_song, scaler


def get_song_features_by_name(df_song, song_name):
    # Search for the song (case-insensitive)
    matched = df_song[df_song['name'].str.lower() == song_name.lower()]
    if matched.empty:
        raise ValueError("Song not found. Please check the song name.")
    # Use the first match if multiple are found
    song_row = matched.iloc[0]
    # Return the features as a numpy array
    return song_row[['acousticness', 'instrumentalness', 'speechiness']].values


def euclidean_distance(user_input, song_features):
    return np.sqrt(np.sum((user_input-song_features)**2))
    

def recommend_songs(X, df_song, acousticness, instrumentalness, speechiness, scaler, k=5):
    #user input processing / standardization
    user_input=np.array([[acousticness, instrumentalness, speechiness]])
    user_input=scaler.transform(user_input)[0]

    distances = []
    for idx, song_features in enumerate(X):
        dist = euclidean_distance(user_input, song_features)
        distances.append((dist, idx))
    
    distances.sort()
    recommendations = distances[:k]
    
    for i, (dist, idx) in enumerate(recommendations, start=1):
        song = df_song.iloc[idx]
        print(f"Recommendation {i}: {song['name']}")

        
if __name__ == "__main__":
    # Train the model and get the scaler
    X, df_song, scaler = train_model()
    
    # Ask user for a song name
    song_name = input("Please enter a song name for recommendations: ")
    
    try:
        # Retrieve the song's feature values from the dataset
        user_song_features = get_song_features_by_name(df_song, song_name)
    except ValueError as e:
        print(e)
        exit()
    
    # Unpack the features
    acousticness, instrumentalness, speechiness = user_song_features
    print("\nRecommendations based on the song:", song_name)
    
    # Use the recommend_songs function to display recommendations
    recommend_songs(X, df_song, acousticness, instrumentalness, speechiness, scaler, k=5)