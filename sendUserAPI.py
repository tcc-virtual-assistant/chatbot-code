import requests

def sendAPI(answer, apiUserid):
    apiUseridPATCH = apiUserid+1
    print(f"API USER ID PATCH: {apiUseridPATCH}")
    info = {'avillaAnswer': answer}
    url = (f'http://127.0.0.1:8000/question/{apiUseridPATCH}')
    requests.patch(url, json=info)
    print('a')      