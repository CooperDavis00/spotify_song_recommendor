# Spotify Song Recommendor

The Spotify Song Recommendor is a Python-based project that leverages machine learning techniques to recommend songs based on user preferences. The system takes a user's liked songs (in CSV format) and uses features like danceability, energy, tempo, and genre to build a personalized recommendation model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Files](#data-files)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CooperDavis00/spotify_song_recommendor.git
Change into the project directory:

bash
Copy
Edit
cd spotify_song_recommendor
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
Activate the virtual environment:

On Windows:

bash
Copy
Edit
.\venv\Scripts\activate
On macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Prepare your input CSV file (liked_songs.csv) containing your liked songs from your Spotify profile. Ensure it follows the required format with the necessary columns (e.g., Track URI, Track Name, Artist Name(s)).

Run the data_loader.py script to load the data and add audio features to your CSV:

bash
Copy
Edit
python data_loader.py
Once the data is processed, you can proceed to train a model and generate recommendations.

Project Structure
bash
Copy
Edit
spotify_song_recommendor/
│
├── data/
│   └── liked.csv        # Raw user data from Spotify profile
│   └── liked_w_features.csv   # Processed data with additional audio features
|   └── LargeSongDataset.csv   # Song Data used to make predictions
│
├── src/
│   ├── data_loader.py         # Script for loading and processing Spotify data
│   ├── model.py               # Script for building recommendation model
│   └── utils.py               # Helper functions for the project
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
Data Files
liked_songs.csv: Raw CSV data containing information about the user's liked songs, including Track URI, Artist Name(s), and Album Details.

liked_w_features.csv: Processed CSV file that includes additional audio features such as danceability, energy, key, and tempo, fetched using the Spotify API.

Dependencies
pandas - for data manipulation and cleaning.

spotipy - for interacting with the Spotify API.

scikit-learn - for building machine learning models (if applicable).

matplotlib - for visualizations (if needed).