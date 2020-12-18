from django.conf.urls import url
from django.urls import path
from . import views

app_name = 'deepfake'

urlpatterns = [
    url('^$', views.IndexView.as_view(), name='index'),
    url('int:pk/', views.UploadView.as_view(), name='uploadpage'),
    path('uploadFile/', views.UploadView.upload_file, name='upload_file')
]