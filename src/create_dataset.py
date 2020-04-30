#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:18:02 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine 

from movie_info import *
from user_prefs import *
from rating_preds import *

from sqlalchemy import create_engine

def calc_sim(user_pref, movie_vec, movie_in_pref, normalize=True):
    '''
    Calculates similarity metric between user's preferences and those same characteristics 
    of the movie making prediction on
    
    user_pref(np array or sparse array): array of user's (genre/cast/crew) preferences
    movie_vec(np array or sparse array): array of the (genre/cast/crew) of the movie that is being compared
    movie_in_pref(boolean): if movie that is being compared is included in user's (genre/cast/crew) preferences
                            e.g. user originally has watched the movie - need to exclude and subtract
                            its movie_genre from genre_pref
    normalize(boolean): min-max normalize user_pref array if True
    '''
    
    # if movie is included in user's aggregate vector, need to remove movie vector
    # before calcuating simlarity metric 
    if movie_in_pref:
        user_pref = user_pref - movie_vec
        
    if normalize:
        # try just using scikitlearn minmax scaler
        user_pref = np.true_divide(user_pref - user_pref.min(), user_pref.max() - user_pref.min())
        
    cosine_sim = 1 - cosine(user_pref, movie_vec)
    
    return cosine_sim


def get_sim(user_id, movie_id, type_cols, df_user_pref, df_movie_info, movie_in_pref):

    '''
    Selects user's genre, cast, or crew vector (based off type_cols) from df_user_pref
    and the movie's genre, cast, or crew vector from df_movie_info and calculates
    the simlarity between the two vectors
    '''    

    user_pref = df_user_pref[df_user_pref['user_id'] == user_id]
    user_pref_vec = user_pref[type_cols].to_numpy()

    movie_info = df_movie_info[df_movie_info['movie_id'] == movie_id]
    movie_info_vec = movie_info[type_cols].to_numpy()     
    
    similarity = calc_sim(user_pref_vec, movie_info_vec, movie_in_pref)
    
    return similarity


def create_dataset(df_user_watched, df_genre_pref, df_cred_pref, df_movie_info):
    '''
    Adds to df_movies_watched the similarity metric of user's genre preferences 
    to genre of movie making prediction on 
    '''
    data = df_user_watched.copy()
    
    genre_cols = ['unknown', 'action', 'adventure', 'animation', 'childrens',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 
                  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
                  'war', 'western']
    genre_sims = []
    
    cast_col = ['cast_count']
    cast_sims = []
    
    crew_col = ['crew_count']
    crew_sims = []
    
    rating_preds = []
    
    # open SQL connection before calling get_user_movie_sims on each row
    engine = create_engine('mysql://root:pw@localhost/recommender')
    with engine.connect() as connection: 
    
        for idx, row in data.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            watched_movie = row['watched_movie']
            
            # if user watched movie, need to remove movie vector from user's vector done in calc_sim
            if watched_movie == 1:
                movie_in_pref = True
            else:
                movie_in_pref = False
                
            # get similarity between user's genre preferences to movie genre
            user_genre_sim = get_sim(user_id, movie_id, genre_cols, df_genre_pref, df_movie_info, movie_in_pref)
            genre_sims.append(user_genre_sim)
            
            # get similarity between user's cast preferences to movie's cast
            user_cast_sim = get_sim(user_id, movie_id, cast_col, df_cred_pref, df_movie_info, movie_in_pref)
            cast_sims.append(user_cast_sim)
            
            # get similarity between user's crew preferences to movie's crew
            user_crew_sim = get_sim(user_id, movie_id, crew_col, df_cred_pref, df_movie_info, movie_in_pref)
            crew_sims.append(user_crew_sim)
            
            # get predicted user rating for movie
            rating_pred = get_rating_pred(user_id, movie_id, connection)
            rating_preds.append(rating_pred)
        
    data['genre_sim'] = genre_sims
    data['cast_sim'] = cast_sims
    data['crew_sim'] = crew_sims
    data['rating_pred'] = rating_preds
    
    return data

if __name__ == '__main__':
    
    df_movie_info = get_movie_info()
    df_user_genre_pref, df_user_cred_pref, df_user_watched = get_user_tables(df_movie_info)
    
    data = create_dataset(df_user_watched, df_user_genre_pref, df_user_cred_pref, df_movie_info)