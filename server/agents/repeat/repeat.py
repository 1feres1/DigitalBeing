import requests
from requests.api import get 
def get_message(message) : 
  
    url = f"http://127.0.0.1:9090//api/search/smart-agent/search/{message}"
    print(url)
    try : 
        r = requests.get(url = url)
        return(r.json()['answer'][0]['message'])
    except Exception as ex : 
        print('intent detection api error ',ex)
print(get_message('hello'))
class Repeat:
        def __init__(self):
            pass

        def handle_message(self,message):
            return get_message(message)