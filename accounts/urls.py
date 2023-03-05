from django.urls import path

from .views import SignUpView, DetailView, DeleteView, UpdateEmailView, ProjectsView

# This file defines the accessible endpoints within the internal app accounts.

app_name = 'accounts'
urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),  # sign up view
    path("<int:pk>/delete/", DeleteView.as_view(), name="delete"),  # delete view of specific user
    path("<int:pk>/detail/", DetailView.as_view(), name="detail"),  # details view of specific user
    path("<int:pk>/detail/email", UpdateEmailView.as_view(), name="update_email"),  # update email view of specific user
    path("<int:pk>/projects/", ProjectsView.as_view(), name="projects"),  # projects view of specific user
]
