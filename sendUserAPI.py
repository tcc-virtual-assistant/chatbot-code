import requests

def sendAPI(answer, apiUserid):
    apiUserid = apiUserid+1
    print(apiUserid)
    url = (f'http://127.0.0.1:8000/question/{apiUserid}')
    r = requests.put(url, data={'avillaAnswer': answer})
    print('foi')