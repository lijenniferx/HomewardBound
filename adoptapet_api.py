import json, requests
  
data = {'apikey' : '987zyx',
  "objectType" : "animals",
  "objectAction" : "define"};
  

'species':'Dog',
    'status':'Available',
    'sex':'Female'}
data_json = json.dumps(data)
payload = {'json_playload': data_json, 'apikey': 'xbJ9QnVs'}

url = 'https://api.rescuegroups.org/rest/?key=xbJ9QnVs&type=orgs&updatedAfter=1190073600'
r = requests.get(url)

requests.get('https://api.rescuegroups.org/rest/?key=xbJ9QnVs&type=orgs&limit=100&name=Weatherford')

    

 plt.plot(data[pd.to_datetime(data.observation_date) > pd.datetime(2007,4,5)], data[pd.to_datetime(data.observation_date) > pd.datetime(2007,4,5)]['UNEMPLOYMENT.1']
