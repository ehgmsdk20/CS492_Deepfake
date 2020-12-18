"""cs320 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import url, include
from django.conf import settings
from django.conf.urls.static import static
from deepfake import views as deepfake_views

urlpatterns = [
    url('admin/', admin.site.urls),
    url('^$', deepfake_views.IndexView.as_view(), name = "root"),
    url('int:pk/', deepfake_views.UploadView.as_view(), name='uploadpage'),
    url('deepfake/', include('deepfake.urls'))
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)