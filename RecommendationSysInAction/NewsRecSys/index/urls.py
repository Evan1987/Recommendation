
from django.conf.urls import url
from .views import home, login, switch_user


urlpatterns = [
    url(r"^home/$", home),
    url(r"^login/$", login),
    url(r"^switch_user/$", switch_user),
]
