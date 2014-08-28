import pandas as pd
import pymysql as mdb
import pickle as pk
from get_model_columns import get_columns

con =  mdb.connect('localhost', 'root','***','HomewardBound'); 
items = pd.read_csv('Dogs_Final_DEMO.csv')


with open('finalmodel.pk','r') as f:
    model  = pk.load(f)

## checking csv database against scraped data, to make sure that it is up to date. 

with open('current_animal_data.pk','r') as f:
    web_data = pk.load(f)   
    
for i in web_data.keys():  
     items.ix[items.AnimalID == int(i),'Fixed']  = web_data[i]['Fixed']
     items.ix[items.AnimalID == int(i),'HasName']  = web_data[i]['HasName']
     
X, junk = get_columns(items)

### generating new dataframes
     
critterfeatures = pd.DataFrame(columns= ['AnimalID','Name','Breed','Age','Size','Sex','Fixed'])
critteroutcomes = pd.DataFrame(columns = ['AnimalID','AdoptionPercentage','ImprovementFix','ImprovementName','WithFix','WithName','WithBoth'])
index = 0
for i in items.AnimalID.tolist(): 
    i = str(i)
    critterfeatures.loc[index] = [int(i),web_data[i]['Name'],web_data[i]['Breed'],web_data[i]['Age'],web_data[i]['Size'],web_data[i]['Sex'],web_data[i]['Fixed']]
    
    if (web_data[i]['Name'] == ''):
        nameimprovement = 1
    else:
        nameimprovement = 0
        
    if (web_data[i]['Fixed'] == 0):
        fiximprovement = 1
    else:
        fiximprovement = 0
        
    critteroutcomes.loc[index] = [int(i),0,fiximprovement,nameimprovement,0,0,0]
    index+=1

         
        
    
### base percentage chance of adoption   

critteroutcomes['AdoptionPercentage'] = model.predict_proba(X)[:,1] * 100
for i in range(len(X)):
    items['Status'].loc[i] = 1 if critteroutcomes['AdoptionPercentage'].loc[i] > 50 else 0

items.to_csv('Dogs_Final_DEMO.csv',index = False)


### unnamed dogs...calculate percentage chance if they are named
name_items = items.copy()
name_items.HasName = 1
X_allnames,junk = get_columns(name_items)
critteroutcomes['WithName'] = model.predict_proba(X_allnames)[:,1] * 100

### unfixed dogs

fix_items = items.copy()
fix_items.Fixed = 1
X_allfix,junk = get_columns(fix_items)
critteroutcomes['WithFix'] = model.predict_proba(X_allfix)[:,1] * 100

### unfixed and unnamed dogs
fixname_items = items.copy()
fixname_items.Fixed = 1
fixname_items.HasName = 1
X_allfixname,junk = get_columns(fixname_items)
critteroutcomes['WithBoth'] = model.predict_proba(X_allfixname)[:,1] * 100



    
with con:
        cur = con.cursor()
        cur.execute("USE HomewardBound")
        cur.execute("DROP TABLE IF EXISTS Demo_Critter_Bio")
        cur.execute("DROP TABLE IF EXISTS Demo_Critter_Predictions")
        critterfeatures.to_sql(con = con, name = 'Demo_Critter_Bio', if_exists = 'replace', flavor = 'mysql')
        critteroutcomes.to_sql(con = con, name = 'Demo_Critter_Predictions', if_exists = 'replace', flavor = 'mysql')

                    
        
