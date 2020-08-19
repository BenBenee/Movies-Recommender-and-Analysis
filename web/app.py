from flask import Flask, request, jsonify, render_template
import pickle
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from difflib import SequenceMatcher

def access_json(data, index):
    result = data
    try:
        for idx in index:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
mv = pd.read_csv('../movies_metadata.csv', parse_dates = ['release_date'], encoding='latin1')
json_cols = ['collection','genres','production_companies','production_countries']
    
for cols in json_cols:
    mv[cols] = mv[cols].apply(json.loads)

cr = pd.read_csv('../credits.csv', encoding='latin1')

cr['cast'] = cr['cast'].apply(json.loads)
    
mv['franchise'] = mv.collection.apply(lambda x: access_json(x, ['name']))
mv['collection_poster'] = mv.collection.apply(lambda x: access_json(x, ['poster_path']))

prod_comp_cols = ['company1','company2']
for i,col in enumerate(prod_comp_cols):
    mv[col] = mv.production_companies.apply(lambda x: access_json(x, [i,'name']))
    
cr['actor_lead'] = cr.cast.apply(lambda x: access_json(x, [0, 'name']))

genre_cols = ['genre1','genre2']
    
for i,col in enumerate(genre_cols):
    mv[col] = mv.genres.apply(lambda x: access_json(x, [i,'name']))

df_movies = pd.merge(mv, cr, left_on='id', right_on='id')

df_movies = df_movies[['id','franchise','poster_path','title','release_date','Director','company1','company2','actor_lead','genre1','genre2','original_language','vote_average','vote_count','overview','tagline']]    

df_movies['poster_path'] = 'https://image.tmdb.org/t/p/w500'+ df_movies['poster_path']

df_movies['year'] = df_movies.release_date.dt.year.fillna(0).astype('int')

list_genre = list(df_movies.genre1.value_counts().index)

vote_df = df_movies.sort_values('vote_count').reset_index().drop(columns='index')
vote_df['vote_count'] = vote_df.vote_count.fillna(0).astype('int')
    
vote_df = vote_df[vote_df.vote_count>=459]

con_rec = df_movies[(df_movies.original_language=='en') & (df_movies.vote_count>=117)].fillna('')

con_rec['description'] = (con_rec['franchise']+' ')*8 + con_rec['tagline'] + (con_rec['overview']+' ')*7+ (con_rec['Director']+' ')*8+(con_rec['company1']+' ')*8+(con_rec['company2']+' ')*8+ (con_rec['actor_lead']+' ')*10+con_rec['genre1']+(con_rec['title']+' ')*5

tf = TfidfVectorizer(analyzer='word',ngram_range=(1,2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(con_rec['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print('--MODEL BUILT--')

con_rec = con_rec.reset_index()
titles = con_rec['title']
ind = pd.Series(con_rec.index,index=con_rec['title'])

def content_recommender(title):
    idx = ind[title]
    if idx.shape != ():
        idx = ind[title].iloc[0]          #I use iloc to choose the first title appear in case of duplicated index
        
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return con_rec.loc[movie_indices,['poster_path','title','vote_average','year']]


print('--DATA LOADED AND PREPARED--')
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/genre', methods=['GET'])
def genre():
    return render_template('genre.html')

@app.route('/content', methods=['GET'])
def content():
    return render_template('content.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    prob_list = []
    
    for idx,genre in enumerate(list_genre):
        prob = SequenceMatcher(None, features[0], genre).ratio()
        prob_list.append((idx,prob))
        
    features[0] = list_genre[sorted(prob_list, key=lambda x: x[1], reverse=True)[0][0]]
    
  
    rec_gen = genre_recommender(features[0],features[1])
    rec_gen = rec_gen.rename(columns={'poster_path':'poster','vote_average':'TMDB Rating'}).drop_duplicates(subset=['title'])
    
    return render_template('genre.html', column_names=rec_gen.columns.values, row_data=list(rec_gen.values.tolist()),  zip=zip)

@app.route('/predict2',methods=['POST'])
def predict2():
    features = [x for x in request.form.values()]
    prob_list = []
    
    for idx,title in enumerate(con_rec['title']):
        prob = SequenceMatcher(None, features[0], title).ratio()
        prob_list.append((idx,prob))
    
    features[0] = con_rec.loc[sorted(prob_list, key=lambda x: x[1], reverse=True)[0][0],'title']
    rec_con = content_recommender(features[0])
    rec_con = rec_con.rename(columns={'poster_path':'poster','vote_average':'TMDB Rating'}).drop_duplicates(subset=['title'])
    
    return render_template('content.html', column_names=rec_con.columns.values, row_data=list(rec_con.values.tolist()),  zip=zip)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict_proba([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    

def genre_recommender(gen,year_=1900):

    global vote_df
    rec_gen = vote_df[(vote_df.genre1==gen) & (vote_df.year>=int(year_))].sort_values('vote_average', ascending=False)
    
    return rec_gen[['poster_path','title','vote_average','year']].reset_index().drop('index',axis=1)

if __name__ == "__main__":

    app.run(debug=True)
    
    
