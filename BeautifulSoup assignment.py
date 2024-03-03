#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import bs4
from bs4 import BeautifulSoup
import requests


# Write a python program to display IMDB’s Top rated 100 Indian movies’ data
# https://www.imdb.com/list/ls056092300/ (i.e. name, rating, year ofrelease) and make data frame.

# In[ ]:


box=soup.find('div',class_='lister-item-content')
title=box.find('h3').get_text()
print(title)


# In[ ]:


url='https://www.imdb.com/list/ls056092300/'
page=requests.get(url)
page


# In[ ]:


soup=BeautifulSoup(page.content)
soup


# In[ ]:


firstT=soup.find('h3',class_="lister-item-header")
firstT.text.replace('\n','')


# In[ ]:





# In[ ]:


titles=[]
headers=soup.find_all('h3',class_="lister-item-header")
for i in headers:
    titles.append(i.text.replace('\n',""))
    
titles


# In[ ]:


Rating=[]
headers=soup.find_all('div',class_="ipl-rating-star small")
for i in headers:
    Rating.append(i.text.replace('\n',""))
    
Rating


# In[ ]:


Yearofrelease=[]
headers=soup.find_all('span',class_="lister-item-year text-muted unbold")
for i in headers:
    Yearofrelease.append(i.text)
    
Yearofrelease


# In[ ]:


import pandas as pd
dic={'movie_Name':titles,'Rating':Rating,'Yearofrelease':Yearofrelease}
df=pd.DataFrame(dic)

df


# 2) Write a python program to scrape product name, price and discounts from https://peachmode.com/search?q=bags

# In[ ]:


url='https://peachmode.com/search?q=bags'
page=requests.get(url)
page
soup=BeautifulSoup(page.content)
soup.prettify()
PName=[]
header=soup.find_all('p',class_="sc-jSUZER fcDsVC NewProductCardstyled__StyledDesktopProductTitle-sc-6y2tys-5 ejhQZU NewProductCardstyled__StyledDesktopProductTitle-sc-6y2tys-5 ejhQZU")
for i in header:
    PName.append(i.text)
PName


# In[ ]:


url='https://peachmode.com/search?q=bags'
page=requests.get(url)
page


# In[ ]:


soup=BeautifulSoup(page.content)
soup.prettify()


# In[ ]:


listp=[]
headers=soup.find_all('span',class_="discounted_price st-money money done")
for i in headers:
    listp.append(i)
listp


# In[ ]:





# Q2-Please visit https://www.keaipublishing.com/en/journals/artificial-intelligence-in-agriculture/most-downloaded-articles/ and scrap-
# 
# a) Paper title
# 
# b) date
# 
# c) Author

# In[ ]:


url=' https://www.keaipublishing.com/en/journals/artificial-intelligence-in-agriculture/most-downloaded-articles/'
page=requests.get(url)
page


# In[ ]:


soup=BeautifulSoup(page.content)
soup


# In[ ]:


Papertitle=[]
headers=soup.find_all('h2',class_="h5 article-title")
for i in headers:
    Papertitle.append(i.text)
Papertitle



# In[ ]:


date=[]
headers=soup.find_all('p',class_="article-date")
for i in headers:
    date.append(i)
date


# In[ ]:


author=[]
headers=soup.find_all('p',class_="article-authors")
for i in headers:
    author.append(i.text)
author


# In[ ]:


import pandas as pd
dic={'Papertitle':Papertitle,'date':date,'author':author}
df=pd.DataFrame(dic)

df


# In[ ]:




