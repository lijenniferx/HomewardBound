import requests, json

github_url = "https://api.rescuegroups.org"
data = json.dumps({'adopted':'status', 'description':'some test repo'}) 
r = requests.post(github_url, data, auth=('xbJ9QnVs'))
print r.json