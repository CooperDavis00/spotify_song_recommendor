# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def preprocess_data(user_df: pd.DataFrame, other_df: pd.DataFrame):
    # Drop unnecessary columns
    other_df = other_df.drop(['msPlayed','time_signature', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms'], axis=1)
    other_df['like'] = 0
    other_df['predicted like'] = 0
    other_df = other_df.dropna(subset=['genre'])
    other_df = pd.get_dummies(other_df, columns=['key', 'mode'])

    # Format user's liked song dataset
    user_df['like'] = 1
    user_df['predicted like'] = 0
    user_df = user_df.drop('Track URI', axis=1)
    user_df.rename(columns={'Track Name': 'trackName', 'Artist Name(s)': 'artistName', 'Genre': 'genre'}, inplace=True)
    user_df = pd.get_dummies(user_df, columns=['key', 'mode'])

    # Combine the datasets
    combined_df = pd.concat([user_df, other_df], ignore_index=True)

    # Feature engineering 
    combined_df['similar_genres'] = (combined_df['genre'] == combined_df['genre']).astype(int)
    sound_attributes = ['danceability', 'energy', 'loudness', 'speechiness', 
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    similarity_matrix = euclidean_distances(combined_df[sound_attributes])
    combined_df['sound_similarity'] = similarity_matrix.sum(axis=0)

    return combined_df, sound_attributes

def split_and_clean_data(df, sound_attributes):
    X = df.drop(['like', 'predicted like', 'similar_genres', 'sound_similarity'], axis=1)
    y_genre = df['similar_genres']
    y_sound = df['sound_similarity']

    X_train, X_val, y_train_genre, y_val_genre, y_train_sound, y_val_sound = train_test_split(
        X, y_genre, y_sound, test_size=0.2, random_state=21, stratify=y_genre)

    # Convert all columns to float and handle NaN
    X_train = np.nan_to_num(X_train.astype(float), nan=0.0, posinf=1e10, neginf=-1e10)
    X_val = np.nan_to_num(X_val.astype(float), nan=0.0, posinf=1e10, neginf=-1e10)

    return X_train, X_val, y_train_genre, y_val_genre, y_train_sound, y_val_sound

def build_genre_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_sound_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
