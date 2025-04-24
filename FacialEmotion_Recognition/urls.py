"""
URL configuration for FacialEmotion_Recognition project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.urls import path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
from app01.views import detect_image, detect_video, detect_realtime, detect_emotion

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", TemplateView.as_view(template_name='index.html'), name='index'),
    path("api/detect_image/", detect_image, name='detect_image'),
    path("api/detect_video/", detect_video, name='detect_video'),
    path("api/detect_realtime/", detect_realtime, name='detect_realtime'),
    path("api/detect-emotion/", detect_emotion, name='detect_emotion'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
