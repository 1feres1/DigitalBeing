import requests 
message = 'hello'
url = f"http://127.0.0.1:9090//api/search/smart-agent/search/{message}"
print(url)
try : 
    r = requests.get(url = url)
    print(r.json())
except Exception as ex : 
    print('intent detection api error ',ex)
