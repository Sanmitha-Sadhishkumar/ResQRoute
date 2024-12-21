import firebase_admin
from firebase_admin import db, credentials

def initialize():
    cred = credentials.Certificate("../credentials.json")
    firebase_admin.initialize_app(cred, {'databaseURL':'https://fire-evacuation-c6bf9-default-rtdb.firebaseio.com/'})