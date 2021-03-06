#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:44:47 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

import os

from scipy.spatial.distance import cosine

from sqlalchemy import create_engine

def create_movie_sims(df_movie_info):
    '''
    Compares every movie to each other based off of genres and creates and exports new SQL table
    with both movie id's and the cosine similarity between the two - does permutation
    instead of combination due to nature of calculating predicted user rating
    '''
    
    cols = ['movie_id', 'unknown', 'action', 'adventure', 'animation', 'childrens',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 
                  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
                  'war', 'western']
    
    df_movie_genres = df_movie_info[cols]
    movie_gen_arr = df_movie_genres.to_numpy()
    
    num_movies = len(movie_gen_arr)
    
    movie_sims = [['movie1_id', 'movie2_id', 'genre_sim']]
    
    for i in range(num_movies):
        for j in range(num_movies):
            
            movie1_id = movie_gen_arr[i][0]
            movie1_genres = movie_gen_arr[i][1:]
            
            movie2_id = movie_gen_arr[j][0]
            movie2_genres = movie_gen_arr[j][1:]
            
            movies_sim = 1 - cosine(movie1_genres, movie2_genres)
            
            movies_sim_arr = [movie1_id, movie2_id, movies_sim]
            movie_sims.append(movies_sim_arr)
            
    df_genre_sim = pd.DataFrame(movie_sims[1:], columns=movie_sims[0])

    make_sql_table(df_genre_sim, 'movie_sims')


def make_sql_table(df, table_name):
    '''
    Helper function to push DataFrame to MySQL table
    '''
    
    pw = os.environ.get("MYSQLPW")
    path = 'mysql://root:{}@localhost/recommender'.format(pw)
    engine = create_engine(path)
    
    with engine.connect() as connection: 
        df.to_sql(con=connection, name=table_name, if_exists='replace', index=False)
    

