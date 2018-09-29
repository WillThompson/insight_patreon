from flask import Flask

app = Flask(__name__)

from applet.patreonpro import views
