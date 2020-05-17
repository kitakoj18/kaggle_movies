#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:54:21 2020

@author: kojikitagawa
"""
import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine 

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
        
    if normalize and (user_pref.max() != user_pref.min()):
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
    
    #if type_cols is either 'cast_count' or 'crew_count'
    if type(type_cols)==str:
        
        user_pref = df_user_pref[df_user_pref['user_id'] == user_id].reset_index(drop=True)
        user_pref_vec = user_pref[type_cols][0].toarray()
    
        movie_info = df_movie_info[df_movie_info['movie_id'] == movie_id].reset_index(drop=True)
        movie_info_vec = movie_info[type_cols][0].toarray()

    #else type_cols is list of genre_cols
    else: 
        user_pref = df_user_pref[df_user_pref['user_id'] == user_id]
        user_pref_vec = user_pref[type_cols].to_numpy()
    
        movie_info = df_movie_info[df_movie_info['movie_id'] == movie_id]
        movie_info_vec = movie_info[type_cols].to_numpy()     
    
    similarity = calc_sim(user_pref_vec, movie_info_vec, movie_in_pref)
    
    return similarity