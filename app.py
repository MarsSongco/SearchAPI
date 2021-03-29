#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Api, Resource
from rank_bm25 import BM25Okapi
import spacy
import pickle
import json


app = Flask(__name__) #create the Flask app
CORS(app)

with open('model.pkl', 'rb') as model_pickle:
    tok_text=pickle.load(model_pickle)
bm25 = BM25Okapi(tok_text)
articles = pd.read_csv('articles.csv')

@app.route('/getSimilar')
def getSimilar():
    query = request.args.get('query')
    tokenized_query = query.lower().split(" ")

    titles = bm25.get_top_n(tokenized_query, articles.title.values, n=10)
    ids = bm25.get_top_n(tokenized_query, articles.id.values, n=10)
    df = pd.DataFrame(list(zip(titles, ids)), columns =['Title', 'ID']) 
    response = df.to_json(orient="records")
    parsed = json.loads(response)
    return json.dumps(parsed)
if __name__ == '__main__':
    app.run(debug=True)


# %%
