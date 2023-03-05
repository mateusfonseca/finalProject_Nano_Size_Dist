from django.contrib.auth.models import User
from django.db import models


# This file defines the models that reflect the database's entities.
# Each class is an entity and their properties are the tables' columns.
# Instances of these classes are the database's entries, the tables' rows.
# The relationship between the entities can be basically expressed as:
# [USER]<--1:N-->[PROJECT]<--1:N-->[ANALYSIS]<--1:4-->[FILE]


# Project entity
# One user owns zero or many projects
# One project belongs to only one user
class Project(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100, null=True, blank=True)
    description = models.CharField(max_length=500, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.title}: {self.description}, by {self.user}.'


# Analysis entity
# One project owns zero or many analyses
# One analysis belongs to only one project
class Analysis(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    title = models.CharField(max_length=100, null=True, blank=True)
    description = models.CharField(max_length=500, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.title}: {self.description}, from {self.project}'


# File entity
# One analysis owns exactly four files:
#   the original image (TIFF), the processed image (PNG),
#   the size distribution graph (PNG) and the dataframe (CSV)
# One file belongs to only one analysis
class File(models.Model):
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE)
    title = models.CharField(max_length=100, default=None, blank=True, null=True)
    uri = models.TextField(max_length=1000, default=None, blank=True, null=True)
    type = models.CharField(max_length=10, default='image', blank=True, null=True)
    source = models.ForeignKey('self', on_delete=models.CASCADE, blank=True, null=True, default=None)

    def __str__(self):
        return f'Title: {self.title}' \
               f'URI: {self.uri}' \
               f'Type: {self.type}' \
               f'Source: {self.source}' \
               f'Analysis: {self.analysis}' \
               f'Project: {self.analysis.project}' \
               f'User: {self.analysis.project.user}'
