from django.urls import path

from . import views

app_name = 'traitement_app'
urlpatterns = [
    path('', views.index, name='index'),
    path('image/', views.image, name='image'),
    path('MobileNet/', views.MobileNet, name='MobileNet'),
    path('ResNet101/', views.ResNet101, name='ResNet101'),
    
]
