# CA1: CRUD Application

from django.urls import path

from . import views
from .views import IndexView

# This file defines the accessible endpoints within the app polls.

app_name = 'nanoer'
urlpatterns = [
    path('', IndexView.as_view(), name='index'),  # app's home view
    # path('', IndexView.post, name='display_image'),
]
