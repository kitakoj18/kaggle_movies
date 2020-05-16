#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:35:16 2020

@author: kojikitagawa
"""
import numpy as np
import pandas as pd

import random

cols = ['movie_id', 'unknown', 'action', 'adventure', 'animation', 'childrens',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 
                  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
                  'war', 'western', 'cast_count', 'crew_count']

def get_user_genres(df_movies_watched):
    '''
    Creates table with sum of genres watched by user
    To look at history of the genres of the movies the user has watched
    '''
    
    df = df_movies_watched.copy()
    df.drop(columns=['movie_id', 'rating', 'timestamp', 'cast_count', 'crew_count'], inplace=True)
    
    df_genre_pref = df.groupby('user_id', sort=False).sum().reset_index()
    
    return df_genre_pref


def get_user_creds(df_movies_watched):
    '''
    Creates table with sum of cast and crew watched by user
    To look at history of the cast and crew of the movies the user has watched
    '''
    
    df = df_movies_watched.copy()
    df = df[['user_id', 'cast_count', 'crew_count']]
    
    user_groups = df.groupby('user_id', sort=False)
    df_cred_pref = pd.concat([user_groups['cast_count'].apply(np.sum), user_groups['crew_count'].apply(np.sum)], axis=1).reset_index()
    
    return df_cred_pref


def generate_unwatched_movies(user_id, user_movie_set, movie_set, num_unwatched):
    '''
    Helper function for user_watched_movies to randomly select num_unwatched number of movies 
    that the user has not watched based off of their set of movies they have watched
    '''
    unwatched_movie_ids = []
    num_movies = len(movie_set)
    
    while num_unwatched > 0: 
        
        # randomly select a movie id from movie_set
        rand_movie_idx = random.randint(0, num_movies-1)
        # if selected movie id is in user_movie_set, user watched movie so continue
        if movie_set[rand_movie_idx] in user_movie_set:
            continue
        
        unwatched_movie_ids.append(movie_set[rand_movie_idx])
        num_unwatched -= 1

    df_unwatched = pd.DataFrame({'movie_id': unwatched_movie_ids})
    df_unwatched['user_id'] = user_id
    df_unwatched['watched'] = 0
    #reorder columns to match column order of dataframe it's being appended to
    df_unwatched = df_unwatched[['user_id', 'movie_id', 'watched']]
    
    return df_unwatched

def get_user_watched_movies(df_movies_watched, movie_set, num_unwatched):
    '''
    Function that generates table with all users and movies they did watch
    and predetermined number of movies they did not watch to use as a base table to create final dataset
    '''

    df_watched = df_movies_watched.copy()
    df_watched = df_watched[['user_id', 'movie_id']]
    # add column to indicate movie was watched by user
    df_watched['watched'] = 1
    
    # get set of movies watched for each user
    df_user_movie_set = df_watched.groupby('user_id', sort=False) \
                                    .agg({'movie_id': lambda x: set(x)}) \
                                    .reset_index()                   
    
    # for each user, select num_watched number of movies NOT watched by user
    for idx, row in df_user_movie_set.iterrows():
        
        user_id = row['user_id']
        user_movie_set = row['movie_id']
        
        # get movies not watched by user and append to df_watched
        df_unwatched = generate_unwatched_movies(user_id, user_movie_set, movie_set, num_unwatched)
        df_watched = df_watched.append(df_unwatched, ignore_index=True)
        
    return df_watched


def get_user_tables(df_movie_info, num_unwatched=5):
    '''
    Function to return all user-related tables: users' genre preferences, users' credit
    preferences, and users' movie watched history
    '''
    
    # narrow movies rated by users to only movies we have information on
    df_ratings = pd.read_csv('../data/movie_lens/ratings_subset.csv')
    df_avail_movies = df_movie_info[cols]
    df_movies_watched = df_ratings.merge(df_avail_movies, left_on='movie_id', right_on='movie_id')
    
    # get set of unique movie_ids from movies we have information on  
    movie_set = df_avail_movies['movie_id'].unique()
    
    # generate user preference tables and return
    df_user_genre_pref = get_user_genres(df_movies_watched)
    df_user_credit_pref = get_user_creds(df_movies_watched)
    df_user_watched = get_user_watched_movies(df_movies_watched, movie_set, num_unwatched)
    
    return df_user_genre_pref, df_user_credit_pref, df_user_watched