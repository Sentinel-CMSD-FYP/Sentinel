from django.urls import path

from . import views

# Listed before apps.clips.urls in config/urls.py: ClipsView answers GET /clips as the clips
# app does and adds POST (upload).
urlpatterns = [
    path("clips", views.ClipsView.as_view()),
    path("clips/<str:clip_id>/runs", views.RunsView.as_view()),
    path("clips/<str:clip_id>/runs/<str:run_id>", views.RunView.as_view()),
    path("runs/active", views.ActiveRunView.as_view()),
    path("streams", views.StreamsView.as_view()),
    path("streams/<str:stream_id>", views.StreamView.as_view()),
    path("streams/<str:stream_id>/stop", views.StreamStopView.as_view()),
    path("streams/<str:stream_id>/samples", views.StreamSamplesView.as_view()),
    path("streams/<str:stream_id>/video", views.stream_video),
]
