# spoti

import pandas as pd
import swifter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from tqdm import tqdm


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import statsmodels
import sklearn


pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)

%matplotlib inline

# Load the required libraries
import pandas as pd
import numpy as np
import json
import re
import os
import glob
import math
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)

%matplotlib inline
DATASETS_PATH = 'Datasets/Spotify_Playlist_Datasets/'

FIELDS = ['pid', 'name', 'description', 'modified_at', 'num_artists',
         'num_albums', 'num_tracks', 'num_followers', 'num_edits',
         'duration_ms', 'collaborative']

# Concatenate all json files into a single dataframe and flaten the json object tracks in the original dataset.
for i, dataset in enumerate(glob.glob(DATASETS_PATH+'/*.json')):

    data = json.load(open(dataset), encoding="utf-8")
    
    df_ = pd.json_normalize(data['playlists'], 'tracks', FIELDS, record_prefix= 'track_', errors='ignore')
        
    if i == 0:
        df = df_.copy()
    else:
        df = pd.concat([df, df_], axis=0, ignore_index=True)
        
   #Edit columns name to be more descriptive and remove track prefix from them

TRACKS_FIELDS = ['track_pos', 'track_artist_name', 'track_track_uri',
                'track_artist_uri', 'track_track_name', 'track_album_uri',
                'track_duration_ms', 'track_album_name']

df= TRACKS_FIELDS = ['track_pos', 'track_artist_name', 'track_track_uri',
                'track_artist_uri', 'track_track_name', 'track_album_uri',
                'track_duration_ms', 'track_album_name']
json_file_1=open("mpd.slice.0-999.json")
json_file_2=open("mpd.slice.1000-1999.json")
json_file_3=open("mpd.slice.2000-2999.json")
json_file_4=open("mpd.slice.3000-3999.json")
data_1=json.load(json_file_1)
data_2=json.load(json_file_2)
data_3=json.load(json_file_3)
data_4=json.load(json_file_4) 
DATASETS_PATH ='Datasets/Spotify_Playlist_Datasets/'

FIELDS = ['pid', 'name', 'description', 'modified_at', 'num_artists',
         'num_albums', 'num_tracks', 'num_followers', 'num_edits',
         'duration_ms', 'collaborative']

# Concatenate all json files into a single dataframe and flaten the json object tracks in the original dataset.
for i, dataset in enumerate(glob.glob(DATASETS_PATH+'/*.json')):

    data = json.load(open(dataset))
    
    df_ = pd.json_normalize(data['playlists'], 'tracks', FIELDS, record_prefix= 'track_', errors='ignore')
        
    if i == 0:
        df = df_.copy()
    else:
        df = pd.concat([df, df_], axis=0, ignore_index=True)
        
TRACKS_FIELDS = ['track_pos', 'track_artist_name', 'track_track_uri',
                'track_artist_uri', 'track_track_name', 'track_album_uri',
                'track_duration_ms', 'track_album_name']
data_tracks_1=pd.DataFrame()
for i in range(len(data_1['playlists'])):
    data_tracks_1=data_tracks_1.append(pd.DataFrame(data_1['playlists'][i]['tracks']))
data_tracks_1=data_tracks_1.reset_index(drop=True)

data_tracks_2=pd.DataFrame()
for i in range(len(data_2['playlists'])):
    data_tracks_2=data_tracks_2.append(pd.DataFrame(data_2['playlists'][i]['tracks']))
data_tracks_2=data_tracks_2.reset_index(drop=True)

data_tracks_3=pd.DataFrame()
for i in range(len(data_3['playlists'])):
    data_tracks_3=data_tracks_3.append(pd.DataFrame(data_3['playlists'][i]['tracks']))
data_tracks_3=data_tracks_3.reset_index(drop=True)

data_tracks_4=pd.DataFrame()
for i in range(len(data_4['playlists'])):
    data_tracks_4=data_tracks_4.append(pd.DataFrame(data_4['playlists'][i]['tracks']))
data_tracks_4=data_tracks_4.reset_index(drop=True)
track_uri_1=pd.DataFrame(data_tracks_1['track_uri'])
track_uri_2=pd.DataFrame(data_tracks_2['track_uri'])
track_uri_3=pd.DataFrame(data_tracks_3['track_uri'])
track_uri_4=pd.DataFrame(data_tracks_4['track_uri'])
DATASETS_PATH = 'Datasets/Spotify_Playlist_Datasets/'

FIELDS = ['pid', 'name', 'description', 'modified_at', 'num_artists',
         'num_albums', 'num_tracks', 'num_followers', 'num_edits',
         'duration_ms', 'collaborative']

# Concatenate all json files into a single dataframe and flaten the json object tracks in the original dataset.
for i, dataset in enumerate(glob.glob(DATASETS_PATH+'/*.json')):

    data = json.load(open(dataset))
    
    df_ = pd.json_normalize(data['playlists'], 'tracks', FIELDS, record_prefix= 'track_', errors='ignore')
        
    if i == 0:
        df = df_.copy()
    else:
        df = pd.concat([df, df_], axis=0, ignore_index=True)
        
TRACKS_FIELDS = ['track_pos', 'track_artist_name', 'track_track_uri',
                'track_artist_uri', 'track_track_name', 'track_album_uri',
                'track_duration_ms', 'track_album_name']
track_uri_4
track_id_1=pd.DataFrame()
track_id_1[['Name','Item','Track_id']] = track_uri_1['track_uri'].str.split(':',expand=True)

track_id_2=pd.DataFrame()
track_id_2[['Name','Item','Track_id']] = track_uri_2['track_uri'].str.split(':',expand=True)

track_id_3=pd.DataFrame()
track_id_3[['Name','Item','Track_id']] = track_uri_3['track_uri'].str.split(':',expand=True)

track_id_4=pd.DataFrame()
track_id_4[['Name','Item','Track_id']] = track_uri_4['track_uri'].str.split(':',expand=True)
track_id_4
df=pd.read_csv('spotify_ dataset.csv')

df.rename(columns={'duration_ms': 'playlist_duration_ms'}, inplace=True)

for column in df.columns:
    if column in TRACKS_FIELDS:
        df.rename(columns={column : column.split('_', maxsplit=1)[1]}, inplace=True)

df.head()
df.index
df.rename(columns={'duration_ms': 'playlist_duration_ms'}, inplace=True)
df.columns
df.tail()
for column in df.columns:
    if column in TRACKS_FIELDS:
        df.rename(columns={column : column.split('_', maxsplit=1)[1]}, inplace=True)
# Create the function to extract the features from track URI
def uri_to_features(uri):

    client_credentials_manager = SpotifyClientCredentials(client_id='60f548ab778645f69ca7fd89ffa1f0f6',   
                                                      client_secret='53eb71b22a1f43018d290125ac2e1c43') 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Extract audio features
    features = sp.audio_features(uri)[0]
    
    # Extract artists features 
    artist = sp.track(uri)["artists"][0]["id"]
    artist_pop = sp.artist(artist)["popularity"]
    artist_genres = sp.artist(artist)["genres"]
    
    # Extract track features
    track_pop = sp.track(uri)["popularity"]
    
    # Add artists and track features into the audio features
    features["artist_pop"] = artist_pop
    if artist_genres:
        features["genres"] = " ".join([re.sub(' ','_',i) for i in artist_genres])
    else:
        features["genres"] = "unknown"
    features["track_pop"] = track_pop
    
    return features
Client_ID = '60f548ab778645f69ca7fd89ffa1f0f6'
Client_Secret = '53eb71b22a1f43018d290125ac2e1c43'
df['acousticness '] = df['acousticness'].apply(lambda x: str(x).split(":")[-1])
df['acousticness'].head()
df.kurtosis()
all_uri = df.danceability.unique().tolist()
df.danceability.nunique()
p1, p2, p3, p4, p5 = all_uri[:2000], all_uri[2000:40000], all_uri[40000:60000], all_uri[60000:80000], all_uri[80000:]
api_features = []
# Extract the features using function
for i in tqdm(p1):
    try:
        api_features.append(uri_to_features(i))
    except:
        continue
 for i in tqdm(p2):
    try:
        api_features.append(uri_to_features(i))
    except:
        continue       







        



