# Django
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from django.shortcuts import render, redirect

from serpwow.google_search_results import GoogleSearchResults
import json

import csv
import re
import nltk
import numpy as np
import math
from time import time

nltk.download('stopwords')


vectorizer = CountVectorizer()

STOPWORDS = stopwords.words('english')
STEMMER = PorterStemmer()

# Constantes con la ponderacion para los datos
VAL_TITLES = 0.1
VAL_ABSTRACTS = 0.9

# Inicio clasificador


def stopWords(book):
    book2 = book
    for word in book:
        if word in STOPWORDS:
            book2.remove(word)
    return book2


def steamingMethod(book):
    return [STEMMER.stem(word) for word in book]


def clean(line):
    clean = [re.sub('[^A-Za-z0-9]+', ' ', i.lower()) for i in line]
    a = [i.split() for i in clean]
    stm = [steamingMethod(i) for i in a]
    return [stopWords(i) for i in stm]


def bagWords(book):
    data = []
    bag = []
    for i in book:
        data.append(' '.join(i))
    x = vectorizer.fit_transform(data)
    for i in x.toarray():
        bag.append(list(i))
    return bag


def pesado(tf):
    pes = []
    for i in tf:
        p = []
        for j in i:
            if j > 0:
                p.append(1 + math.log10(j))
            else:
                p.append(0)
        pes.append(p)
    return pes


def documetFrecuency(pes):
    df = [0] * len(pes[0])
    for i in pes:
        cont = 0
        for j in i:
            if j > 0:
                df[cont] += 1
            cont += 1
    return df


def idf(book, data):
    idf = []
    n = len(data)
    for i in book:
        idf.append(math.log10(n / i))
    return idf


def tf_idf(td, id):
    final = []
    for i in td:
        val = []
        for j, k in zip(i, id):
            val.append(j * k)
        final.append(val)
    return final


def normalizacion(v):
    nor = []
    for i in v:
        vec = []
        a = 0
        for j in i:
            a = a + j ** 2
        a = math.sqrt(a)
        for k in i:
            vec.append(k / a)
        nor.append(vec)
    return nor


def cos(v1, v2):
    a = 0
    for i, j in zip(v1, v2):
        a += (i * j)
    return a


def jaccardDistance(book, threshold):
    val = np.zeros((len(book), len(book)), dtype=float)
    cont1 = 0
    for i in book:
        cont2 = 0
        for j in book:
            val[cont1][cont2] = (len(np.intersect1d(i, j)) /
                                 len(np.union1d(i, j))) * threshold
            cont2 += 1
        cont1 += 1
    return val


def cosineDistance(book, threshold):
    val = np.zeros((len(book), len(book)), dtype=float)
    cont1 = 0
    for i in book:
        cont2 = 0
        for j in book:
            val[cont1][cont2] = (cos(i, j)) * threshold
            cont2 += 1
        cont1 += 1
    return val


def matriz_distancias(d1, d2):
    matriz3 = np.zeros((len(d1), len(d1)), dtype=float)
    for i in range(len(d1)):
        for j in range(len(d1)):
            matriz3[i][j] += d1[i][j] + d2[i][j]
    return matriz3

# Fin clasificador


def getBooks(query):
    serpwow = GoogleSearchResults("3470FC62D66A40CAA60630032A14BDD8")

    params = {
        "q": query,
        "search_type": "scholar",
        "sort_by": "date",
        "hl": "en",
        "time_period": "last_year"
    }

    result = serpwow.get_json(params)
    return result['scholar_results']


def home(request):
    return render(request, 'index.html')


def result(request):
    # Listas donde se guardaran los datos del csv
    titles = []
    abstracts = []
    q = request.POST['query']
    data = getBooks(q)
    for i in data:
        i['title'] = i['title'].replace('[PDF][PDF]', '')
        i['title'] = i['title'].replace('[BOOK][B]', '')
        i['title'] = i['title'].replace('[HTML][HTML]', '')

    for i in data:
        titles.append(i['title'])
        abstracts.append(i['snippet'])

    dis_tit = jaccardDistance(clean(titles), VAL_TITLES)
    pe = pesado(bagWords(clean(abstracts)))
    doc_fre = documetFrecuency(pe)
    val_idf = idf(doc_fre, pe)
    val_tfidf = tf_idf(pe, val_idf)
    no = normalizacion(val_tfidf)
    dis_abst = cosineDistance(no, VAL_ABSTRACTS)
    final = matriz_distancias(dis_tit, dis_abst)

    
    return render(request, 'result.html', {'data': data})
