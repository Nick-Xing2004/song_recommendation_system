import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#predict and recommend songs based on song's danceability, energy, and duration (measured in ms)

def train_model():  
    df_song=pd.read_csv('./song_recommendation_system/data.csv')   #read in data
    X=df_song[['danceability', 'energy', 'duration_ms']]
    X=X.to_numpy()
    return X, df_song

def euclidean_distance(user_input, song_features):
    return np.sqrt(np.sum((user_input-song_features)**2))

def recommend_songs(X, df_song, danceability, energy, duration_ms, k=5):
    user_input=np.array([danceability, energy, duration_ms])
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
    X, df_song=train_model()
    user_input=input("Please enter song danceability, energy, and time duration(in ms) for recommendations: ")
    danceability, energy, duration_ms = map(float, user_input.strip().split())
    recommend_songs(X, df_song, danceability, energy, duration_ms)




#feature standardization
