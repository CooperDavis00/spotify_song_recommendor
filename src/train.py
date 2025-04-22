from model import preprocess_data, split_and_clean_data, build_genre_model, build_sound_model
import pandas as pd

# Load datasets
user_df = pd.read_csv('data/liked_w_features.csv')
other_df = pd.read_csv('data/Spotify_Song_Attributes.csv')

# Preprocess and get data
combined_df, sound_attributes = preprocess_data(user_df, other_df)
X_train, X_val, y_train_genre, y_val_genre, y_train_sound, y_val_sound = split_and_clean_data(combined_df, sound_attributes)

# Train genre model
genre_model = build_genre_model(X_train.shape[1])
genre_model.fit(X_train, y_train_genre, epochs=10, batch_size=32, validation_data=(X_val, y_val_genre))

# Train sound model
sound_model = build_sound_model(X_train.shape[1])
sound_model.fit(X_train, y_train_sound, epochs=10, batch_size=32, validation_data=(X_val, y_val_sound))

# Optionally save models
genre_model.save('models/genre_model.h5')
sound_model.save('models/sound_model.h5')
