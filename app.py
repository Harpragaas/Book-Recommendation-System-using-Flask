import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from contextlib import suppress
import pickle

books_final = pd.read_pickle('data.pkl')

book_sparse = csr_matrix(books_final)

model = NearestNeighbors(metric = "cosine", algorithm='brute')
model.fit(book_sparse)


def recc(m):
    
    
    distances, suggestions = model.kneighbors(books_final.loc[m].values.reshape(1,-1),5)
    
    recommendations = []
    for i in range(0, len(distances.flatten())):
        if i > 0:
            book_name = books_final.index[suggestions.flatten()[i]]
            #print(book_name)
            recommendations.append(book_name)

    return recommendations



app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend',methods=['GET','POST'])
def recommend():
    book = request.args.get('book')
    recommendation = recc(book)
    return render_template('recommend.html',book=book,recommendation=recommendation,t='h')



    


    

if __name__=="__main__":
    app.run(debug=True)