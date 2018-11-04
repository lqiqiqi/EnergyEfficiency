from django.urls import path

from . import views

urlpatterns = [
    path('', views.input, name = 'calculate'),
    path('result/', views.calculate, name = 'result')
]