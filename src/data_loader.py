import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API authentication
SPOTIPY_CLIENT_ID = 'your_client_id'
SPOTIPY_CLIENT_SECRET = 'your_client_secret'

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

def fetch_audio_features(track_uri):
    """Fetch audio features for a given track URI."""
    audio_features = sp.audio_features(track_uri)[0]  # Get audio features
    if audio_features is None:
        return {key: 'N/A' for key in audio_features.keys()}
    return audio_features

def fetch_track_genres(artists):
    """Fetch genres for a list of artists."""
    genres = []
    for artist in artists:
        artist_info = sp.artist(artist['id'])
        artist_genres = artist_info['genres'] if artist_info['genres'] else ['N/A']
        genres.extend(artist_genres)
    return ', '.join(genres)

def process_data(input_csv, output_csv):
    """Load the input CSV, fetch features and genres, and save the processed data."""
    df = pd.read_csv(input_csv)
    
    updated_rows = []
    
    for index, row in df.iterrows():
        track_uri = row['Track URI']
        track_name = row['Track Name']
        artist_names = row['Artist Name(s)'].split(', ')  # Split artist names by commas
        
        try:
            track_details = sp.track(track_uri)
            artists = track_details['artists']
            genres = fetch_track_genres(artists)
            audio_features = fetch_audio_features(track_uri)

            track_info = {
                'Track URI': track_uri,
                'Track Name': track_name,
                'Artist Name(s)': ', '.join(artist_names),
                'genre': genres,
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'key': audio_features['key'],
                'loudness': audio_features['loudness'],
                'mode': audio_features['mode'],
                'speechiness': audio_features['speechiness'],
                'acousticness': audio_features['acousticness'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'valence': audio_features['valence'],
                'tempo': audio_features['tempo']
            }

            updated_rows.append(track_info)
        
        except Exception as e:
            print(f"Error processing {track_name}: {e}")
            updated_rows.append({
                'Track URI': track_uri,
                'Track Name': track_name,
                'Artist Name(s)': ', '.join(artist_names),
                'genre': 'N/A',
                'danceability': 'N/A',
                'energy': 'N/A',
                'key': 'N/A',
                'loudness': 'N/A',
                'mode': 'N/A',
                'speechiness': 'N/A',
                'acousticness': 'N/A',
                'instrumentalness': 'N/A',
                'liveness': 'N/A',
                'valence': 'N/A',
                'tempo': 'N/A'
            })

    # Create DataFrame and save to CSV
    updated_df = pd.DataFrame(updated_rows)
    updated_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Processed data saved to: {output_csv}")

# Run the data processing
process_data('data/liked_songs.csv', 'data/liked_w_features.csv')