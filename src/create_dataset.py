#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:18:02 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

from movie_info import *
from user_movie_sim import *
from user_prefs import *
from rating_preds import *

from movie_sims import *
from user_pref_dist import *

from sqlalchemy import create_engine

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
    
    cast_col = 'cast_count'
    cast_sims = []
    
    crew_col = 'crew_count'
    crew_sims = []
    
    rating_preds = []
    genre_ents = []
    cast_ents = []
    crew_ents = []
    
    # open SQL connection before calling get_user_movie_sims on each row
    engine = create_engine('mysql://root:pw@localhost/recommender')
    with engine.connect() as connection: 
    
        for idx, row in data.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            watched_movie = row['watched']
            
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
            
            # get user's genre entropy value
            genre_ent = get_entropy(user_id, 'user_genre_ent', connection)
            genre_ents.append(genre_ent)
            
            cast_ent = get_entropy(user_id, 'user_cast_ent', connection)
            cast_ents.append(cast_ent)
            
            crew_ent = get_entropy(user_id, 'user_crew_ent', connection)
            crew_ents.append(crew_ent)
        
    data['genre_sim'] = genre_sims
    data['cast_sim'] = cast_sims
    data['crew_sim'] = crew_sims
    data['rating_pred'] = rating_preds
    data['genre_ent'] = genre_ents
    data['cast_ent'] = cast_ents
    data['crew_ent'] = crew_ents
    
    df_users = pd.read_csv('../data/movie_lens/users.csv')
    df_users = df_users[['user_id', 'age', 'gender', 'occupation']]
    
    data = data.merge(df_users, left_on='user_id', right_on='user_id')
    
    # fillna in case user_pref vector is zero and sim calculation is NaN
    data.fillna(0, inplace=True)
    
    return data

if __name__ == '__main__':
    
    df_movie_info = get_movie_info()
    df_user_genre_pref, df_user_cred_pref, df_user_watched = get_user_tables(df_movie_info)
    
    #generate_movie_sims(df_movie_info)
    create_movie_sims(df_movie_info)
    create_ratings_table(df_movie_info)
    
    create_user_entropy(df_user_genre_pref, 'user_genre_ent')
    create_user_entropy(df_user_cred_pref, 'user_cast_ent')
    create_user_entropy(df_user_cred_pref, 'user_crew_ent')
    
    data = create_dataset(df_user_watched, df_user_genre_pref, df_user_cred_pref, df_movie_info)