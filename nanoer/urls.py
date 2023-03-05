from django.urls import path

from .views import IndexView, CreateView, DetailView, temp_clean_up, get_project_list, save_to_project, DeleteView, \
    delete_confirm, download_analysis, AnalysisUpdateView, AnalysisDeleteView, ProjectUpdateView, download_project

# This file defines the accessible endpoints within the app nanoer.

app_name = 'nanoer'
urlpatterns = [
    path('', IndexView.as_view(), name='index'),  # app's home view
    path('save_to_project', save_to_project, name='save_to_project'),  # save analysis to project
    path('project/create/<int:pk>', CreateView.as_view(), name='create'),  # create new project view
    path('project/<int:pk>', DetailView.as_view(), name='detail'),  # details view of specific project
    path('temp_clean_up', temp_clean_up, name='temp_clean_up'),  # delete temporary files
    path('get_project_list', get_project_list, name='get_project_list'),  # list all projects of specific user
    path('project/<int:pk>/delete', DeleteView.as_view(), name='delete'),  # delete view of specific project
    # delete view of specific analysis
    path('analysis/<int:pk>/delete', AnalysisDeleteView.as_view(), name='delete_analysis'),
    path('<str:model>/<int:pk>/delete_confirm', delete_confirm, name='delete_confirm'),  # confirm model deletion
    path('analysis/<int:pk>/download', download_analysis, name='download_analysis'),  # download analysis' files
    path('project/<int:pk>/download', download_project, name='download_project'),  # download project's files
    # update view of specific analysis
    path('analysis/<int:pk>/edit', AnalysisUpdateView.as_view(), name='edit_analysis'),
    # update view of specific project
    path('project/<int:pk>/edit', ProjectUpdateView.as_view(), name='edit_project'),
]
