from django.urls import path

from . import views

urlpatterns = [
    path("health", views.HealthView.as_view()),
    path("clips", views.ClipListView.as_view()),
    path("clips/<str:clip_id>", views.ClipDetailView.as_view()),
    path("clips/<str:clip_id>/video", views.video),
    path("clips/<str:clip_id>/layers/<str:layer>", views.LayerView.as_view()),
]
