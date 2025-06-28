import pandas as pd
import numpy as np
import json

# Read the dataset
df = pd.read_csv('dataset.csv')
df.columns = df.columns.str.strip()

# Convert duration from ms to minutes
df['duration_min'] = df['duration_ms'] / 60000

# Ensure we have all the required columns
required_columns = [
    'track_id', 'artists', 'album_name', 'track_name', 'track_genre',
    'popularity', 'duration_min', 'explicit', 'danceability', 'energy',
    'loudness', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

# Create a smart sample of 500 tracks
# First, get a list of at least 10 different genres
top_genres = df['track_genre'].value_counts().nlargest(15).index.tolist()

# Initialize an empty sample dataframe
sample_df = pd.DataFrame()

# For each genre, sample tracks across the popularity spectrum
for genre in top_genres:
    genre_df = df[df['track_genre'] == genre]
    
    # Skip if less than 10 tracks in this genre
    if len(genre_df) < 10:
        continue
    
    # Divide the popularity range into bins and sample from each
    genre_df['popularity_bin'] = pd.qcut(genre_df['popularity'], 
                                         q=min(5, len(genre_df.popularity.unique())), 
                                         duplicates='drop')
    
    # Sample from each bin - try to get about 30-35 tracks per genre
    bin_sample_size = max(2, min(7, len(genre_df) // 5))
    genre_sample = genre_df.groupby('popularity_bin').apply(
        lambda x: x.sample(min(bin_sample_size, len(x)))
    ).reset_index(drop=True)
    
    # Add explicit and clean tracks
    explicit_tracks = genre_df[genre_df['explicit'] == True].sample(min(5, sum(genre_df['explicit'])))
    clean_tracks = genre_df[genre_df['explicit'] == False].sample(min(5, sum(~genre_df['explicit'])))
    
    # Combine and remove duplicates
    genre_sample = pd.concat([genre_sample, explicit_tracks, clean_tracks]).drop_duplicates('track_id')
    
    # Add to the main sample
    sample_df = pd.concat([sample_df, genre_sample])

# If we don't have 500 tracks yet, add more tracks randomly
if len(sample_df) < 500:
    remaining_tracks = df[~df['track_id'].isin(sample_df['track_id'])].sample(500 - len(sample_df))
    sample_df = pd.concat([sample_df, remaining_tracks])

# If we have more than 500 tracks, trim down to 500
if len(sample_df) > 500:
    sample_df = sample_df.sample(500)

# Select only required columns and reset index
sample_df = sample_df[required_columns].reset_index(drop=True)

# Convert to JSON
sample_json = sample_df.to_json(orient='records')

# Store the JSON in a variable that we'll insert into the HTML file
with open('sample_data.json', 'w') as f:
    f.write(sample_json)

print(f"Sampled {len(sample_df)} tracks across {sample_df['track_genre'].nunique()} genres")
print(f"Explicit tracks: {sum(sample_df['explicit'])}")
print(f"Clean tracks: {sum(~sample_df['explicit'])}")
print(f"Sample saved to sample_data.json")

# Display some stats about the sample
print("\nGenre distribution:")
print(sample_df['track_genre'].value_counts().head(10))

print("\nPopularity distribution:")
print(sample_df['popularity'].describe())

print("\nLoudness distribution:")
print(sample_df['loudness'].describe())
