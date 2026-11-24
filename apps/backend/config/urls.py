from django.urls import include, path

urlpatterns = [
    path("api/v1/", include("apps.pipeline.urls")),  # upload + runs; GET /clips stays the same
    path("api/v1/", include("apps.clips.urls")),
]
