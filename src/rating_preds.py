#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:16:50 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd


def get_rating_pred(user_id, movie_id, conn):
    '''
    Get rating prediction based off user's past ratings and genre simlarity to 
    movie making prediction on
    '''
    
    # get all of user's past ratings, excluding movie making prediction on
    # (in the case where watched=1) and the similarity of these rated movies
    # to the movie making prediction on based off of genres
    query = ''' 
            SELECT r.rating, s.genre_sim
            FROM ratings r
            JOIN movie_sims s
                ON r.movie_id = s.movie1_id
            WHERE r.user_id = {}
                AND r.movie_id != {}
                AND s.movie2_id = {}
            '''.format(user_id, movie_id, movie_id)
            
    user_rating_movie_sims = pd.read_sql_query(query, conn)
    
    rating_pred = calc_rating_pred(user_rating_movie_sims)
    return rating_pred


def calc_rating_pred(user_rating_movie_sims):
    '''
    Calculates user's rating prediction as the sum of user's past ratings 
    weighted by similarity to movie making prediction on
    '''
    
    rating_arr = user_rating_movie_sims['rating'].to_numpy()
    genre_sims_arr = user_rating_movie_sims['genre_sim'].to_numpy()
    
    # if no movies are similar to movie making prediction on, return rating of 0
    if genre_sims_arr.sum() == 0:
        return 0
    
    else: 
        # weighted sum
        rating_pred = np.true_divide((rating_arr*genre_sims_arr).sum(), genre_sims_arr.sum())
        return rating_pred
        
    