
from django.conf.urls import url
from .views import one, cates


urlpatterns = [
    url(r"^one/$", one),
    url(r"^cates/$", cates),
]
