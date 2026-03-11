from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('crop-predict/', views.predict_crop, name='predict_crop'),
    path('disease-predict/', views.predict_disease, name='predict_disease'),
    path('yield-predict/', views.predict_yield, name='predict_yield'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/detect-disease/', views.detect_disease, name='detect_disease'),
    path('predict-fertilizer/', views.predict_fertilizer, name='predict_fertilizer'),
]