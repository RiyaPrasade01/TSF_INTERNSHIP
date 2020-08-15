import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Hours'=6})

print(r.json())