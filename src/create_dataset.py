#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:18:02 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine
    

def get_user_genres(df_movies_watched):
    '''
    Creates table with sum of genres watched by user
    To look at history of the genres of the movies the user has watched
    '''
    df = df_movies_watched.copy()
    df.drop(columns=['movie_id', 'rating', 'timestamp'], inplace=True)
    
    df_genre_pref = df.groupby('user_id', sort=False).sum().reset_index()
    
    return df_genre_pref
    

def calc_genre_sim(genre_pref, movie_genre, movie_in_pref=True, normalize=True):
    '''
    Calculates similarity of user's genre preferences to genre of movie making prediction on
    
    genre_pref(np array): array of user's genre preferences
    movie_genre(np array): array of the genre of the movie that is being compared
    movie_in_pref(boolean): if movie that is being compared is included in user's genre preferences
                            i.e. user has watched the movie - need to exclude and subtract
                            movie_genre from genre_pref
    normalize(boolean): min-max normalize genre_pref array if True
    '''
    
    if movie_in_pref:
        genre_pref = genre_pref - movie_genre
        
    if normalize:
        # try just using scikitlearn minmax scaler
        genre_pref = np.true_divide(genre_pref - genre_pref.min(), genre_pref.max() - genre_pref.min())
    
    return cosine(genre_pref, movie_genre)
        

# df_movies_watched should be named to data and this DataFrame should only contain 
# user_id, movie_id, and whether user watched the movie
def add_genre_sim(df_movies_watched, df_genre_pref, df_movies):
    '''
    Adds to df_movies_watched the similarity metric of user's genre preferences 
    to genre of movie making prediction on 
    '''
    data = df_movies_watched.copy()
    similarities = []
    genres = ['unknown', 'action', 'adventure', 'animation', 'childrens',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 
                  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
                  'war', 'western']
    
    for idx, row in df_movies_watched.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        #watched_movie = row['watched_movie']
        
        # get user's genre preference from df_genre
        user_pref = df_genre_pref[df_genre_pref['user_id'] == user_id]
        # get genre vector
        genre_pref = user_pref[genres].to_numpy()
        
        # get movie genre from df_movies
        movie_genre = df_movies[df_movies['movie_id'] == movie_id]
        # get genre vector
        movie_genre_vec = movie_genre[genres].to_numpy()
        
        # here add condition: if watched_movie = 1 then movie_in_pref = True
        # else if watched_movie = 0 then movie_in_pref = False
        genre_sim = calc_genre_sim(genre_pref, movie_genre_vec)
        similarities.append(genre_sim)
        
    data['genre_sims'] = similarities
    return data
    