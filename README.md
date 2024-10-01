# Spotify Music Recommendation System 🎶

## Overview
This project is a Spotify Music Recommendation System that leverages clustering and machine learning techniques to recommend songs based on user input. Whether you want to find similar tracks by a specific artist or discover more music in your favorite genre, this system provides personalized suggestions.

## Features
- **Genre-based Recommendations**: Suggests songs based on selected genres.
- **Artist-based Recommendations**: Recommends music from similar artists.
- **Song-based Recommendations**: Finds songs similar to your favorite tracks.
- **Data Visualization**: Visualizes important features like genres, clusters, and trends using Seaborn, Plotly, and Yellowbrick.

## Project Flow
1. **Data Preprocessing**: Datasets were collected and cleaned.
2. **Clustering**: K-Means clustering was used to group similar songs.
3. **Visualization**: Various visualizations were implemented to better understand the data.
4. **Recommendations**: Personalized recommendations based on song features, artist, year, and genres.

## Prerequisites
To run this project, you'll need:
- Python 3.x
- Jupyter Notebook or any preferred Python IDE
- Libraries: 
  - `pandas`, `numpy`, `seaborn`, `plotly`, `matplotlib`
  - `scikit-learn` for clustering and preprocessing
  - `spotipy` for Spotify API integration
  - `dotenv` for managing environment variables

You can install the required Python packages using:
```sh
pip install -r requirements.txt
