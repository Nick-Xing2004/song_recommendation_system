import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#predict and recommend songs based on song's danceability, energy, and duration (measured in ms)

def train_model():  
    df_song=pd.read_csv('./song_recommendation_system/data.csv')   #read in data
    X=df_song[['danceability', 'energy', 'duration_ms']]
    X=X.to_numpy()

    #standardize the features
    scaler=StandardScaler()
    X_standardized=scaler.fit_transform(X)
    return X_standardized, df_song, scaler

def get_song_features_by_names(song_name, df_song):
    # Search for the song (case-insensitive)
    matched = df_song[df_song['name'].str.lower() == song_name.lower()]
    if matched.empty:
        raise ValueError("Song not found. Please check the song name.")
    # Use the first match if multiple are found
    song_row = matched.iloc[0]
    # Return the features as a numpy array
    return song_row[['danceability', 'energy', 'duration_ms']].values     #returned as numpy arr 


def euclidean_distance(user_input, song_features):
    return np.sqrt(np.sum((user_input-song_features)**2))
    

def recommend_songs(X, df_song, danceability, energy, duration_ms, scaler, k=5):
    #user input processing / standardization
    user_input=np.array([[danceability, energy, duration_ms]])
    user_input=scaler.transform(user_input)[0]

    distances=[]     #initialize a list for holding obtained euclidean distances
    for idx, song_features in enumerate(X):
        dist=euclidean_distance(user_input, song_features)  
        distances.append((dist, idx))
    
    distances.sort()  
    recommendations=distances[:k]     #obtain the top k recommendations from the training set
    i=1
    for dist, idx in recommendations:
        song=df_song.iloc[idx]
        print(f'recommendation{i}: {song['name']}')
        i+=1

if __name__ == "__main__":
    X, df_song, scaler=train_model()        #scaler used for further user input standardization
    
    while True:
        try:
            song_name=input("Please enter song name to get song recommendations: ")
            user_song_features=get_song_features_by_names(song_name, df_song)      #extract the corresponding song features based on the song's name from the df
            break
        except ValueError as e:
            print(f'{e} Please try again.\n')
        

    danceability, energy, duration_ms=user_song_features

    recommend_songs(X, df_song, danceability, energy, duration_ms, scaler)