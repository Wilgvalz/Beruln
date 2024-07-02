import requests

url = "http://127.0.0.1:8000/":
params = {"input_parameters": ""}

response = requests.post(url, params=params)

print(response.text)
