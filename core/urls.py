from django.urls import path
from . import views

APP_NAME = 'core'

urlpatterns = [
    path('', views.home_view, name="home_view"),
]
