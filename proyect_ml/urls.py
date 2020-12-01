"""proyect_ml URLs module."""
from django.contrib import admin
from django.urls import path
from documents import views as documents_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', documents_views.home),
    path('search/', documents_views.result, name="search"),
]
