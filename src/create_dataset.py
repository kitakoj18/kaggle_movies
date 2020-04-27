#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:18:02 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine 

from joins import *

def calc_sim(user_pref, movie_vec, movie_in_pref=True, normalize=True):
    '''
    Calculates similarity of user's genre preferences to genre of movie making prediction on
    
    user_pref(np array or sparse array): array of user's (genre/cast/crew) preferences
    movie_vec(np array or sparse array): array of the (genre/cast/crew) of the movie that is being compared
    movie_in_pref(boolean): if movie that is being compared is included in user's (genre/cast/crew) preferences
                            e.g. user has watched the movie - need to exclude and subtract
                            movie_genre from genre_pref
    normalize(boolean): min-max normalize user_pref array if True
    '''
    
    if movie_in_pref:
        user_pref = user_pref - movie_vec
        
    if normalize:
        # try just using scikitlearn minmax scaler
        user_pref = np.true_divide(user_pref - user_pref.min(), user_pref.max() - user_pref.min())
    
    return cosine(user_pref, movie_vec)
        

# df_movies_watched should be named to data and this DataFrame should only contain 
# user_id, movie_id, and whether user watched the movie
def add_genre_sim(df_movies_watched, df_genre_pref, df_movies, df_cred_pref, df_creds):
    '''
    Adds to df_movies_watched the similarity metric of user's genre preferences 
    to genre of movie making prediction on 
    '''
    data = df_movies_watched.copy()
    genre_sims = []
    genres = ['unknown', 'action', 'adventure', 'animation', 'childrens',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 
                  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
                  'war', 'western']
    
    cast_sims = []
    crew_sims = []
    
    for idx, row in df_movies_watched.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        #watched_movie = row['watched_movie']
        
        # get user's genre preference from df_genre_pref
        user_genre_pref = df_genre_pref[df_genre_pref['user_id'] == user_id]
        # genre vector
        genre_pref = user_genre_pref[genres].to_numpy()
        # get movie's genre from df_movies
        movie_genre = df_movies[df_movies['movie_id'] == movie_id]
        # movie's genre vector
        movie_genre_vec = movie_genre[genres].to_numpy()
        # here add condition: if watched_movie = 1 then movie_in_pref = True
        # else if watched_movie = 0 then movie_in_pref = False
        genre_sim = calc_sim(genre_pref, movie_genre_vec)
        genre_sims.append(genre_sim)
        
        # get user's cast preference from df_cred_pref
        user_cast_pref = df_cred_pref[df_cred_pref['user_id'] == user_id]
        # cast vector
        cast_pref = user_cast_pref[['cast_count']].to_numpy()
        # get movie's cast from df_creds
        movie_cast = df_creds[df_movies['movie_id'] == movie_id]
        movie_cast_vec = movie_cast[['cast_count']]
        # get similarity of user cast preferences to movie's cast
        cast_sim = calc_sim(cast_pref, movie_cast_vec)
        cast_sims.append(cast_sim)
        
        # get user's crew preference from df_cred_pref
        user_crew_pref = df_cred_pref[df_cred_pref['user_id'] == user_id]
        # crew vector
        crew_pref = user_crew_pref[['crew_count']].to_numpy()
        # get movie's cast from df_creds
        movie_crew = df_creds[df_movies['movie_id'] == movie_id]
        movie_crew_vec = movie_crew[['crew_count']]
        # get similarity of user crew preferences to movie's crew
        crew_sim = calc_sim(crew_pref, movie_crew_vec)
        crew_sims.append(crew_sim)
        
        
    data['genre_sims'] = genre_sims
    data['cast_sims'] = cast_sims
    data['crew_sims'] = crew_sims
    
    return data
    
if __name__ == '__main__':
    
    df_movie_info = get_movie_info()
    