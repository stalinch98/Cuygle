# Django
from django.shortcuts import render, redirect

from serpwow.google_search_results import GoogleSearchResults
import json


def getBooks():
    serpwow = GoogleSearchResults("BB5814F4D0CA4B85A0E59711D4B0F1FB")

    params = {
        "q": "machine learning",
        "search_type": "scholar",
        "hl": "en"
    }

    result = serpwow.get_json(params)
    return result['scholar_results']


def home(request):
    return render(request, 'index.html')


def result(request):
    data = getBooks()
    for i in data:
        i['title'] = i['title'].replace('[PDF][PDF]', '')
        i['title'] = i['title'].replace('[BOOK][B]', '')
        
    return render(request, 'result.html', {'data': data})
