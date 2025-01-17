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
import transformers
import torch

from time import time

nltk.download('stopwords')


vectorizer = CountVectorizer()

STOPWORDS = stopwords.words('english')
STEMMER = PorterStemmer()

# Constantes con la ponderacion para los datos
VAL_TITLES = 0.1
VAL_ABSTRACTS = 0.9

# Inicio Gpt2
gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-large')
gpt_model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-large')


def gen_text(prompt_text, tokenizer, model, n_seqs=1, max_length=125):
    encoded_prompt = tokenizer.encode(
        prompt_text, add_special_tokens=False, return_tensors="pt")
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length+len(encoded_prompt),
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=n_seqs
    )
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    generated_sequences = []
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence)
        total_sequence = (
            prompt_text +
            text[len(tokenizer.decode(encoded_prompt[0],
                                      clean_up_tokenization_spaces=True, )):]
        )
        generated_sequences.append(total_sequence)
    return generated_sequences
# Fin Gpt2

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
    serpwow = GoogleSearchResults("3157272BDD884C99A66ABDDCAABA66C1")

    params = {
        "q": query,
        "search_type": "scholar",
        "sort_by": "relevance",
        "time_period": "last_year",
        "hl": "en",
        "gl": "us"
    }

    result = serpwow.get_json(params)
    return result['scholar_results']


def top(dis_mt):
    top = {}
    val_final = []
    for i in range(0, len(dis_mt)):
        cor = dis_mt[i]
        for j in range(0, len(dis_mt[i])):
            top[j] = cor[j]
        top2 = sorted(top.items(), reverse=True, key=lambda x: x[1])
        top_final = [i[0] for i in top2]
        top_final = top_final[1:4]
        top_final2 = [i[1] for i in top2]
        top_final2 = top_final2[1:4]
        dic = {}
        for k in range(0, len(top_final)):
            dic['top_{}'.format(k + 1)] = (top_final[k]+1)
            dic['val_{}'.format(k + 1)] = round(top_final2[k]*100, 2)
        dic['val'] = (i+1)
        val_final.append(dic)
    return val_final


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
    similar_top = top(final)

    # Inicio consulta documento
    titles.append(q)
    abstracts.append(q)
    dis_tit2 = jaccardDistance(clean(titles), VAL_TITLES)
    pe2 = pesado(bagWords(clean(abstracts)))
    doc_fre2 = documetFrecuency(pe2)
    val_idf2 = idf(doc_fre2, pe2)
    val_tfidf2 = tf_idf(pe2, val_idf2)
    no2 = normalizacion(val_tfidf2)
    dis_abst2 = cosineDistance(no2, VAL_ABSTRACTS)
    final2 = matriz_distancias(dis_tit2, dis_abst2)
    valTop = final2[10]
    top_doc = {}
    for j in range(0, len(valTop)):
        top_doc[j] = round(valTop[j] * 100, 2)
    top2 = sorted(top_doc.items(), reverse=True, key=lambda x: x[1])
    top_final = [i[0] for i in top2]
    top_final = top_final[1:11]
    top_final2 = [i[1] for i in top2]
    top_final2 = top_final2[1:11]
    data_final = []
    for i in top_final:
        data_final.append(data[i])
    for i, j in zip(data_final, top_final2):
        i['por'] = j
    # Fin consulta documento

    return render(request, 'result.html', {'data': data_final, 'similar_top': similar_top})


def gpt_view(request):
    com_body = request.GET.get('com_body', None)
    val = gen_text(com_body, gpt_tokenizer, gpt_model, max_length=50)
    val_total = val[0].replace(com_body, '')
    abs_gp = {
        'val': str(val_total).capitalize()
    }
    gpt2_final = []
    gpt2_final.append(abs_gp)
    return render(request, 'ajax_gpt2.html', {'abs_gp': gpt2_final})
