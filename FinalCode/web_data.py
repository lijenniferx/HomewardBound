from bs4 import BeautifulSoup
import string
import urllib2
import re
from urllib import urlretrieve
import numpy as np
import os
import pickle as pk

current_url = 'http://www.adoptapet.com/shelter75721-dogs.html'
page = urllib2.urlopen(current_url)
soup = BeautifulSoup(page)

#direction = 'arrow right'
#while direction == 'arrow right':
    ### get all animals
#all_dog_pics = soup.findAll('img', attrs = {'src' : re.compile('https://'), 'width' : 200})
all_dog_links = soup.findAll('a', attrs = {'class' : "smaller_line_height", 'href' : re.compile('http://www.adoptapet.com/pet/[0-9]{8}-weatherford')})    
all_dog_links = np.unique(map(lambda(x):x['href'], all_dog_links))
    


    #try:
    #    next_link = soup.findAll('a', attrs = {'href' : re.compile('http://www.adoptapet.com/cgi-bin/public/petsearch.cgi/search_pets_by_shelter')})[-1]
    #    direction = next_link.find('img')['alt']
    #    page = urllib2.urlopen(next_link['href'])
    #    soup = BeautifulSoup(page)
    #except:
    #    direction = 'no more links'
        


    
    


allanimalinfo = {}
animaldict ={}
animal_pics = []
index = 1
for animal in all_dog_links:

    animal_page = urllib2.urlopen(animal)
    animalsoup = BeautifulSoup(animal_page)
    
    animal_pics.append(animalsoup.find('meta', attrs = {'property' : 'og:image'})['content'])
    
    animalinfosoup = animalsoup.find('div', attrs = {'class' : 'blue_highlight no_margin'})
    
    animalinfoname = re.findall('[A-Za-z]+!', animalsoup.find('h1', attrs = {'class' : 'museo500'}).text)
    
    animalinfo = {}
    for li in animalinfosoup.findAll('li'):
        if li.find('b'):
            if li.text.split(':')[0] == 'Color':
                animalinfo[li.text.split(':')[0]] = li.text.split(':')[1].strip().split('/')[0]
            else:
                animalinfo[li.text.split(':')[0]] = li.text.split(':')[1].strip()
            
    animalinfo['Fixed'] = 1 if re.search('already spayed|already neutered', animalsoup.text) else 0
    animalinfo['HasName'] = 1 if re.search('My name is [A-z]+[a-z]+', animalsoup.text) else 0
    animalinfo['Name'] = '' if len(animalinfoname) == 0 else animalinfoname[0]
    
    allanimalinfo[animalinfo['ID#']] = animalinfo
    animaldict[index] = animalinfo['ID#']
    index=index + 1
    print index


### saving available animal IDs to disk
with open('list_of_current_animals.pk','w') as f:
    pk.dump([int(i) for i in allanimalinfo.keys()],f)

with open('current_animal_data.pk','w') as f:
    pk.dump(allanimalinfo,f)   
    
####      

index = 1
for animal in animal_pics:
    filename = animaldict[index]
    outpath = os.path.join('/Users/jenniferli/Desktop/Python/Flask/app/static/PetPics', filename)

    urlretrieve(animal, outpath)

    index+=1