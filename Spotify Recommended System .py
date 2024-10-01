#!/usr/bin/env python
# coding: utf-8
Importing Libraries
# In[1]:


import os
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials 
from spotipy.oauth2 import SpotifyOAuth

Quick look at the Dataset 
# In[2]:


data = pd.read_csv("Ruru/data.csv")
data.head(5)


# In[3]:


genre = pd.read_csv("Ruru/data_by_genres.csv")
genre.head(5)


# In[4]:


year = pd.read_csv("Ruru/data_by_year.csv")
year.head(5)


# In[5]:


data.info()


# In[6]:


genre.info()


# In[7]:


year.info()


# Visualization of Dataset 

# In[8]:


from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness','energy','danceability','instrumentalness','liveness',
                 'loudness','speechiness','tempo','valence','duration_ms','explicit','key','mode','year']
X,y = data[feature_names],data['popularity']

features = np.array(feature_names)

visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X,y)
visualizer.show()


# In[10]:


def get_decade(year):
    period = int(year/10)*10
    decade = '{}s'.format(period)
    return decade
data['decade'] = data['year'].apply(get_decade)

sns.set(rc={'figure.figsize':(11,6)})
sns.countplot(x='decade',data=data)
plt.show()


# In[9]:


sound_features = ['acousticness','danceability','energy','instrumentalness','liveness','valence']
fig=px.line(year,x='year',y=sound_features)
fig.show()


# In[12]:


top_genres = genre.nlargest(10, 'popularity')

fig=px.bar(top_genres,x='genres',y=['valence','energy','danceability','acousticness'],barmode='group')
fig.show()

Importing KMeans Libraries
# In[10]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 

cluster_pipeline = Pipeline([('scaler', StandardScaler()),('kmeans',KMeans(n_clusters=10,random_state=42,n_init='auto',algorithm='elkan'))])
X = genre.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre['cluster'] = cluster_pipeline.predict(X) 


# In[ ]:


Visualization of the Data using Kmean


# In[11]:


from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()),('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x','y'], data=genre_embedding)
projection['genres'] = genre['genres']
projection['cluster'] = genre['cluster']

fig = px.scatter(projection, x='x',y='y', color = 'cluster', hover_data=['x','y','genres'])
fig.show()


# In[12]:


cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False, n_init=10))],verbose=False)
X = data.select_dtypes(include=np.number)
number_cols = list(X.columns)
cluster_pipeline.fit(X)
cluster_labels=cluster_pipeline.predict(X)
data['cluster_label'] = cluster_labels


# In[13]:


from sklearn.decomposition import PCA

pca_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=2))])
song_embedding= pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x','y'], data = song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(projection, x='x',y='y', color='cluster', hover_data=['x','y','title'])
fig.show()


# In[17]:


pip install python-dotenv

importing Spotify API and Writing function for music recommendation 
# In[34]:


import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))
    
def find_music(name):
    music_data=defualtdict()
    redult=sp.search(q = 'track: {}'.format(name),limit=1)
    if  results['tracks']['items']==[]:
        return None
    results = results['items']['tracks'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    
    music_data['name']=[name]
    #music_data['year']=[year]
    music_data['explicit']=[int(results['explicit'])]
    music_data['duration_ms']=[results['duration_ms']]
    music_data['popularity']=[results['popularity']]
    
    for key, value in audio_features.items():
        music_data[key]=value
    
    
    return pd.DataFrame(music_data)


# In[35]:


from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence','year','acousticness','danceability','duration_ms','energy','explicit','instrumentalness','key','liveness','loudness','mode','popularity','speechiness','tempo']

def get_music(song,spotify_data):
    try:
        music_data = spotify_data[(spotify_data['name'] == song['name'])
                                 &(spotify_data['year'] == song['year'])].iloc[0]
        return music_data
                                  
    except IndexError:
        return find_song(song['name'],song['year'])

def mean_vector(song_list,spotify_data):
    
    song_vectors =  []
            
    for song in song_list:
        music_data = get_music(song,spotify_data)
        if music_data is None:
           print('Warning : {} does Not exit in Spotify or database'.format(song['name']))
           continue 
        music_vector =  music_data[number_cols].values
        music_vector.append(music_vector)
    
    music_matrix = np.array(list(music_vectors))
    return np.mean(music_matrix, axis=0)


# In[36]:


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def mean_vector(song_list, spotify_data):
    music_vectors = []
    for song in song_list:
        music_data = spotify_data[(spotify_data['name'].str.lower() == song['name'].lower())]
        if music_data.empty:
            continue
        music_vector = music_data[number_cols].values[0]  # Extract feature vector
        music_vectors.append(music_vector)
    music_matrix = np.array(music_vectors)
    return np.mean(music_matrix, axis=0)



def recommend_music(spotify_data, song_title=None, artist=None, year=None, genre=None, n_songs=10):
    
    filtered_data = spotify_data.copy()

    if song_title and isinstance(song_title, str):
        filtered_data = filtered_data[filtered_data['name'].str.lower() == song_title.lower()]

    if artist and isinstance(artist, str):
        filtered_data = filtered_data[filtered_data['artists'].apply(lambda x: artist.lower() in [a.lower() for a in eval(x)])]

    if year and isinstance(year, int):
        filtered_data = filtered_data[filtered_data['year'] == year]

    if genre and isinstance(genre, str) and 'genre' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['genre'].str.lower() == genre.lower()]

    if filtered_data.empty:
        return []

    # Select only the numeric features used during training
    numeric_features = filtered_data[number_cols]  # Assuming `number_cols` is a list of features used during fitting

    scaler = cluster_pipeline.steps[0][1]  # Assuming scaler is from your pipeline
    scaled_data = scaler.transform(numeric_features)
    cluster_labels = cluster_pipeline.steps[1][1].predict(scaled_data)

    recommendations = filtered_data.copy()
    recommendations['cluster'] = cluster_labels
    chosen_cluster = cluster_labels[0]  # Choose the cluster of the first match

    recommendations = recommendations[recommendations['cluster'] == chosen_cluster]

    recommendations = recommendations.sort_values('popularity', ascending=False).head(n_songs)

    columns_to_return = ['name', 'artists', 'year', 'popularity']
    if 'genre' in recommendations.columns:
        columns_to_return.append('genre')

    return recommendations[columns_to_return].to_dict(orient='records')

Importing HTML and Display for final recommendation of music
# In[37]:


from IPython.display import display, HTML

def display_recommendations(recommendations):
    html_content = """
    <div style="background-color:#191414; color: white; padding: 20px; font-family: 'Arial', sans-serif; border-radius: 10px;">
        <h2 style="text-align: center; color: #1DB954;">Spotify Song Recommendations</h2>
        <ul style="list-style-type: none; padding: 0;">
    """
    for song in recommendations:
        html_content += f"""
        <li style='margin: 10px 0; padding: 15px; background-color: #282828; border-radius: 8px; display: flex; align-items: center;'>
            <div style='flex-grow: 1;'>
                <strong style='font-size: 18px;'>{song['name']}</strong> 
                <span style='color: #b3b3b3;'>by {song['artists']}</span>
            </div>
            <div style='text-align: right;'>
                <span style='color: #1DB954; font-size: 12px;'>Spotify</span>
            </div>
        </li>
        """
    html_content += "</ul></div>"

    
    display(HTML(html_content))


recommended_songs = recommend_music(data, song_title='', artist='eminem')#(,year=) 
display_recommendations(recommended_songs)


# In[ ]:




