import pandas as pd
a = pd.read_csv('Pets.csv')
a=a[a['Animal Type'] =='DOG']
#a.groupby('Status').count()


#### impact of arrival date
a['Arrival Date'] = pd.to_datetime(pd.Series(a['Arrival Date']))
a['Arrival Date'] = a['Arrival Date'].apply(lambda(x):x.year)
a.groupby('Arrival Date').apply(lambda(x):sum(x['Status'] =='ADOPTED')/len(x['Status'])).plot(color = 'r',marker = 'o')
a.groupby('Arrival Date').apply(lambda(x):sum(x['Status'] =='EUTHANIZED')/len(x['Status'])).plot(color = 'k',marker = 'o')
plt.ylabel('Probability')
plt.legend(['Adopted','Euthanized'],'best')

#### impact of name
animals_with_name  = a.ix[a['Pet Name'].dropna().index]
animals_without_name = a.ix[np.setxor1d(list(a.index),list(a['Pet Name'].dropna().index))]

adopt_name = len(animals_with_name[animals_with_name['Status']=='ADOPTED'])/len(animals_with_name)
adopt_noname = len(animals_without_name[animals_without_name['Status']=='ADOPTED'])/len(animals_without_name)
plt.bar([1,2], [adopt_name] + [adopt_noname])
plt.ylim([0,0.7])
plt.ylabel('Prob of Adoption')
my_xticks = ['Named', 'Anonymous']
plt.xticks(np.arange(1.4,3.4), my_xticks,rotation=0)

#### impact of age
age_group = a.groupby(['Age','Status']).size()
adopted_age = []
for i in ['BABY','YOUNG ADULT','ADULT','SENIOR']:
    adopted_age+= [age_group[i]['ADOPTED']/sum(age_group[i])]
    
euthanized_age = []
for i in ['BABY','YOUNG ADULT','ADULT','SENIOR']:
    euthanized_age+= [age_group[i]['EUTHANIZED']/sum(age_group[i])]

fig,axes = plt.subplots(nrows=1)
axes.bar(range(len(adopted_age)), np.array(adopted_age) / np.array(euthanized_age),color = 'k')
axes.set_ylabel('Prob(Adopted) / Prob(Euthanized)')

#fig,axes = plt.subplots(nrows = 2)
#axes[0].bar(range(len(adopted_age)), adopted_age,color = 'r')
#axes[0].set_ylabel('Prob of Adoption')
#axes[0].get_xaxis().set_visible(False)
#axes[1].bar(range(len(euthanized_age)), euthanized_age, color = 'k')
#axes[1].set_ylabel('Prob of Euthanasia')
my_xticks = ['BABY','YOUNG ADULT','ADULT','SENIOR']
plt.xticks(np.arange(0.4,4.4), my_xticks,rotation=0)
####################
#### impact of size
#####################
size_group = a.groupby(['Size','Status']).size()
adopted_size = []
for i in ['TOY','SMALL','MEDIUM','LARGE', 'GIANT']:
    adopted_size+= [size_group[i]['ADOPTED']/sum(size_group[i])]
    
euthanized_size = []
for i in ['TOY','SMALL','MEDIUM','LARGE', 'GIANT']:
    euthanized_size+= [size_group[i]['EUTHANIZED']/sum(size_group[i])]

fig,axes = plt.subplots(nrows=1)
axes.bar(range(len(adopted_size)), np.array(adopted_size) / np.array(euthanized_size),color = 'k')
axes.set_ylabel('Prob(Adopted) / Prob(Euthanized)')
#axes[0].set_ylabel('Prob of Adoption')
#axes[0].get_xaxis().set_visible(False)
#axes[1].bar(range(len(euthanized_size)), euthanized_size, color = 'k')
#axes[1].set_ylabel('Prob of Euthanasia')
my_xticks = ['TOY','SMALL','MEDIUM','LARGE', 'GIANT']
plt.xticks(np.arange(0.4,5.4), my_xticks,rotation=0)
plt.axhline(1.0, color = 'r',linestyle = '--')



#### impact of breed
breed_group = a.groupby(['Animal Type','Breed','Status']).size()

## cats
#size_group = breed_group['CAT']
#adopted_breed = []
#for i in breed_group['CAT'].index.levels[0]:
#    try:
#        adopted_breed+= [size_group[i]['ADOPTED']/sum(size_group[i])]
#    except:
#        adopted_breed+=[np.nan]  
#
#euthanized_breed = []
#for i in breed_group['CAT'].index.levels[0]:
#    try:
#        euthanized_breed+= [size_group[i]['EUTHANIZED']/sum(size_group[i])]
#    except:
#        euthanized_breed+=[np.nan]    
#
#cat_diff = np.array(euthanized_breed) - np.array(adopted_breed) # difference in probability
#cat_diff_index = np.argsort(cat_diff)
#bad_cat_breed = cat_diff_index[np.where(cat_diff[cat_diff_index[0:15]] > 0.20)]
#good_cat_breed = cat_diff_index[np.where(cat_diff[cat_diff_index[0:15]] < -0.20)]
#
#breed_group['CAT'].index.levels[0][bad_cat_breed]
#breed_group['CAT'].index.levels[0][good_cat_breed]

# dogs
size_group = breed_group['DOG']
adopted_breed = []
for i in breed_group['DOG'].index.levels[0]:
    try:
        adopted_breed+= [size_group[i]['ADOPTED']/sum(size_group[i])]
    except:
        adopted_breed+=[np.nan]  

euthanized_breed = []
for i in breed_group['DOG'].index.levels[0]:
    try:
        euthanized_breed+= [size_group[i]['EUTHANIZED']/sum(size_group[i])]
    except:
        euthanized_breed+=[np.nan]    

response_diff = np.log(np.array(adopted_breed) / (np.array(euthanized_breed) + 0.001)) # difference in probability
response_diff_index = np.argsort(response_diff)
good_breed = response_diff_index[np.where(response_diff[response_diff_index] > np.log(2))]
bad_breed = response_diff_index[np.where(response_diff[response_diff_index] < np.log(0.5))]

#breed_group['DOG'].index.levels[0][bad_breed]
#breed_group['DOG'].index.levels[0][good_breed]
plt.plot(range(len(list(good_breed))),list(np.flipud(response_diff[good_breed])),'ro')
plt.plot(len(good_breed) + np.array(range(len(bad_breed))),list(np.flipud(response_diff[bad_breed])),'ko')
myticks = np.hstack((np.flipud(breed_group['DOG'].index.levels[0][good_breed]) , np.flipud(breed_group['DOG'].index.levels[0][bad_breed])))
plt.xticks(range(len(list(bad_breed) + list(good_breed))), myticks,rotation=90)
plt.gcf().subplots_adjust(bottom=0.40)
plt.ylabel('log(Prob(Adopted) / Prob(Euthanized))')
plt.xlim([-1,47])
plt.axvline(len(good_breed)-0.5, color = 'k',linestyle = '--')


### impact of color
x = a.groupby('Primary Color').apply(lambda(x):sum(x['Status'] =='ADOPTED')/len(x['Status']))
y = a.groupby('Primary Color').apply(lambda(x):sum(x['Status'] =='EUTHANIZED')/len(x['Status']))