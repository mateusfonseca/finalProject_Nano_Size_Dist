from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.hashers import make_password
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views import generic

from nanoer.models import Project, Analysis


# This file defines all the views in the internal app accounts.
# The views are classes and functions that respond to web requests with appropriate web responses.
# They invoke the templates that will be rendered in return (if applicable) and handle any errors
# that may arise during the handling of the requests.


# function to test whether the user making the request is the same one whose details are being requested
def test_func(self):
    return self.request.user.id == self.kwargs['pk']  # true or false


class SignUpView(generic.CreateView):  # sign up view
    model = User  # User instance from models defined by Django
    form_class = UserCreationForm  # UserCreationForm instance from models defined by Django
    success_url = reverse_lazy("login")  # redirect to login page if sing up is successful
    template_name = "registration/signup.html"  # template to be rendered

    def post(self, request, *args, **kwargs):  # handles POST requests
        if request.method == 'POST':
            form = UserCreationForm(request.POST or None)
            if form.is_valid():
                self.model = User.objects.create()  # create an entry in the database for this new user
                self.model.username = request.POST['username']  # set its username from POST
                self.model.email = request.POST['email']  # set its email address from POST
                # set its password from POST, make_password function hashes it, so it's not plain text
                self.model.password = make_password(request.POST['password1'])
                self.model.save()  # save it to database

                # redirects to login page upon successful user creation
                return redirect(self.success_url)
            else:
                fail = 1 if "This field is required" in str(form.errors) else 2
                return render(request, self.template_name, {'form': UserCreationForm(), 'fail': fail})
        else:
            return render(request, self.template_name, {'form': UserCreationForm()})


# details view of specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class DetailView(LoginRequiredMixin, UserPassesTestMixin, generic.DetailView):
    model = User  # User instance from models defined by Django
    template_name = 'accounts/detail.html'  # template to be rendered

    # tests if user making the request is the same one whose details are being requested
    def test_func(self): return test_func(self)


# delete view of specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class DeleteView(LoginRequiredMixin, UserPassesTestMixin, generic.DeleteView):
    model = User  # User instance from models defined by Django
    template_name = 'accounts/delete.html'  # template to be rendered

    def post(self, request, *args, **kwargs):  # handles POST requests
        self.model = User.objects.get(pk=kwargs['pk'])  # retrieves user from database by id
        if self.model.check_password(request.POST['password']):  # if password matches
            self.model.delete()  # deletes user from database
            return render(request, 'accounts/delete_confirmation.html')  # renders confirmation template
        else:  # if password does not match
            # re-render delete view with error message
            return render(request, self.template_name, context={'fail': 1})

    # tests if user making the request is the same one whose details are being requested
    def test_func(self):
        return test_func(self)


# update email view of specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class UpdateEmailView(LoginRequiredMixin, UserPassesTestMixin, generic.UpdateView):
    model = User  # User instance from models defined by Django
    template_name = 'accounts/change_email.html'  # template to be rendered
    fields = []  # form fields, if applicable

    def post(self, request, *args, **kwargs):  # handles POST requests
        self.model = User.objects.get(pk=kwargs['pk'])  # retrieves user from database by id
        self.model.email = request.POST['email']  # sets its new password
        self.model.save()  # saves to database
        # renders account details view for specific user
        return render(request, 'accounts/detail.html', context={'user': self.model})

    # tests if user making the request is the same one whose details are being requested
    def test_func(self): return test_func(self)


# projects view of specific user
# only logged-in users can access this view
# only users who pass test_func can access this view
class ProjectsView(LoginRequiredMixin, UserPassesTestMixin, generic.ListView):
    model = User  # User instance from models defined by Django
    template_name = 'accounts/projects.html'  # template to be rendered
    context_object_name = 'context'  # context variable to be accessed by the template

    def get_queryset(self):  # returns a list of created projects and a list of created analyses
        # list of projects ordered by most recently updated first
        project_list = Project.objects.filter(user_id=self.kwargs['pk']).order_by('-updated_at')
        # list of analysis that belong to anyone of the projects in project_list
        analysis_list = Analysis.objects.filter(project_id__in=[project.id for project in project_list])

        return {'project_list': project_list, 'analysis_list': analysis_list}  # context

    # tests if user making the request is the same one whose details are being requested
    def test_func(self):
        return test_func(self)
