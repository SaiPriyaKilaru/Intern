#!/usr/bin/env python
# coding: utf-8

# Question 1- Write a Python program to replace all occurrences of a space, comma, or dot with a colon.
# Sample Text- 'Python Exercises, PHP exercises.'
# Expected Output: Python:Exercises::PHP:exercises:
# 

# In[1]:


import regex as re


# In[2]:


st='Python Exercises, PHP exercises.'
result=re.sub(r'[ ,.]',':',st)
print(result)


# Question 2-  Create a dataframe using the dictionary below and remove everything (commas (,), !, XXXX, ;, etc.) from the columns except words.
# Dictionary- {'SUMMARY' : ['hello, world!', 'XXXXX test', '123four, five:; six...']}
# 

# In[13]:


import pandas as pd
import re
s={'SUMMARY' : ['hello, world!', 'XXXXX test', '123four, five:; six...']}
st=pd.DataFrame(s)
st['SUMMARY']=st['SUMMARY'].apply(lambda x:re.sub(r'[\W|\d|X]',"",x))
print(st)


# Question 3- Create a function in python to find all words that are at least 4 characters long in a string. The use of the re.compile() method is mandatory.

# In[29]:


string='Create a function in python to find all words that are at least 4 characters long in a string.'
pattern=re.compile(r"\w{4,}")
r=pattern.findall(string)
print(r)


# Question 4- Create a function in python to find all three, four, and five character words in a string. The use of the re.compile() method is mandatory.

# In[31]:


string='Create a function in python to find all words that are at least 4 characters long in a string.'
pattern=re.compile(r"\b\w{3,5}\b")
r=pattern.findall(string)
print(r)


# Question 5- Create a function in Python to remove the parenthesis in a list of strings. The use of the re.compile() method is mandatory.
# Sample Text: ["example (.com)", "hr@fliprobo (.com)", "github (.com)", "Hello (Data Science World)", "Data (Scientist)"]
# Expected Output:
# example.com
# hr@fliprobo.com
# github.com
# Hello Data Science World
# 

# In[35]:


Text=["example (.com)", "hr@fliprobo (.com)", "github (.com)", "Hello (Data Science World)", "Data (Scientist)"]
for msg in Text:
    match=re.sub(r'[(|)]','',msg)
    print(match)


# Question 6- Write a python program to remove the parenthesis area from the text stored in the text file using Regular Expression.
# Sample Text: ["example (.com)", "hr@fliprobo (.com)", "github (.com)", "Hello (Data Science World)", "Data (Scientist)"]
# Expected Output: ["example", "hr@fliprobo", "github", "Hello", "Data"]
# Note- Store given sample text in the text file and then to remove the parenthesis area from the text.
# 

# In[ ]:





# Question 7- Write a regular expression in Python to split a string into uppercase letters.
# Sample text: “ImportanceOfRegularExpressionsInPython”
# Expected Output: [‘Importance’, ‘Of’, ‘Regular’, ‘Expression’, ‘In’, ‘Python’]
# 

# In[64]:


text="ImportanceOfRegularExpressionsInPython"
result=re.findall(r'[A-Z][a-z]*',text)
print(result)


# Question 8- Create a function in python to insert spaces between words starting with numbers.
# Sample Text: “RegularExpression1IsAn2ImportantTopic3InPython"
# Expected Output: RegularExpression 1IsAn 2ImportantTopic 3InPython
# 

# In[142]:


import re
def insertionOfSpaces(text):
    
    result = re.sub(r'(\D)(\d)', r'\1 \2', text)
    return result

input = "RegularExpression1IsAn2ImportantTopic3InPython"

output = insertionOfSpaces(input)

print(output)


# Question 9- Create a function in python to insert spaces between words starting with capital letters or with numbers.
# Sample Text: “RegularExpression1IsAn2ImportantTopic3InPython"
# Expected Output:  RegularExpression 1 IsAn 2 ImportantTopic 3 InPython
# 

# In[140]:


Text='RegularExpression1IsAn2ImportantTopic3InPython'
result = re.sub(r'(\d+)', r' \1 ', Text)
print(result)


# Question 10- Use the github link below to read the data and create a dataframe. After creating the dataframe extract the first 6 letters of each country and store in the dataframe under a new column called first_five_letters.
# Github Link-  https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv
# 

# In[55]:


df=pd.read_csv('regex.csv')
df


# In[56]:


import pandas as pd
import re


# In[57]:


df['FirstFiveLetters'] = df['Country'].str.extract(r'^(\w{6})')
df


# Question 11- Write a Python program to match a string that contains only upper and lowercase letters, numbers, and underscores.

# In[90]:


pattern=('[a-zA-Z0-9_]')
string='$aW1k_#RKH+_T1542'
result=re.match(pattern,string)
if result:
        print(f'The string  matches the pattern.')
else:
        print(f'The string does not match the pattern.')



# Question 12- Write a Python program where a string will start with a specific number. 

# In[2]:


import re

input_string = "123ABC"
specific_number = 123
pattern = re.compile(fr'{str(specific_number)}')
status= bool(pattern.search(input_string))
if status:
    print(f'The string "{input_string}" starts with the number {specific_number}.')
else:
    print(f'The string "{input_string}" does not start with the number {specific_number}.')


# Question 13- Write a Python program to remove leading zeros from an IP address

# In[1]:


import re
ip_address = "192.012.056.001"
pattern = re.compile(r'\b0+(\d+)\b')
cleaned_ip = pattern.sub(r'\1', ip_address)

print(f'Original IP: {ip_address}')
print(f'Cleaned IP:  {cleaned_ip}')


# Question 14- Write a regular expression in python to match a date string in the form of Month name followed by day number and year stored in a text file.
# Sample text :  ' On August 15th 1947 that India was declared independent from British colonialism, and the reins of control were handed over to the leaders of the Country’.
# Expected Output- August 15th 1947
# Note- Store given sample text in the text file and then extract the date string asked format.
# 

# In[3]:


import re
with open("question14.txt") as file:
    for line in file:
        urls=re.findall('[A-za-z]+\s\w+\s\d{4}',line)
        print(urls)


# Question 15- Write a Python program to search some literals strings in a string. 
# Sample text : 'The quick brown fox jumps over the lazy dog.'
# Searched words : 'fox', 'dog', 'horse'
# 

# In[8]:


Sample_text='The quick brown fox jumps over the lazy dog.'
Searched_words=['fox', 'dog', 'horse']
for i in Searched_words:
    if re.search(i,Sample_text):
        print('Matched!')
    else:
        print('Not Matched!')


# In[17]:


Sample_text='The quick brown fox jumps over the lazy dog.'
Searched_words=['fox', 'dog', 'horse']
for i in Searched_words:
    result=re.search(i,Sample_text)
    print(result)


# Question 16- Write a Python program to search a literals string in a string and also find the location within the original string where the pattern occurs
# Sample text : 'The quick brown fox jumps over the lazy dog.'
# Searched words : 'fox'
# 

# In[ ]:


import warnings
warnings. filterwarnings('ignore')
Sample_text='The quick brown fox jumps over the lazy dog.'
Searched_words=['fox', 'dog', 'horse']
for i in Searched_words:
    result=re.search(i,Sample_text)
    print(result)
    res=Sample_text.index(i)
    print(res)


# 
# Question 17- Write a Python program to find the substrings within a string.
# Sample text : 'Python exercises, PHP exercises, C# exercises'
# Pattern : 'exercises'.
# 

# In[20]:


text='Python exercises, PHP exercises, C# exercises' 
Pattern='exercises'
for i in text:
    result=re.findall(Pattern,text)
print(result)


# 
# Question 18- Write a Python program to find the occurrence and position of the substrings within a string.
# 

# In[34]:


start_pos=0
text='Python exercises, PHP exercises, C# exercises' 
Pattern='exercises'
while start_pos != -1:
        start_pos = text.find(Pattern, start_pos + 1)
        print(start_pos)


# Question 19- Write a Python program to convert a date of yyyy-mm-dd format to dd-mm-yyyy format.

# In[39]:


a=(input("enter date"))
result = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\3-\1-\2', a)
print(result)


# Question 20- Create a function in python to find all decimal numbers with a precision of 1 or 2 in a string. The use of the re.compile() method is mandatory.
# Sample Text: "01.12 0132.123 2.31875 145.8 3.01 27.25 0.25"
# Expected Output: ['01.12', '145.8', '3.01', '27.25', '0.25']
# 

# In[45]:


pattern=re.compile(r'\b\d+\.\d{1,2}\b')
Text="01.12 0132.123 2.31875 145.8 3.01 27.25 0.25"
result=pattern.findall(Text)
print(result)


# Question 21- Write a Python program to separate and print the numbers and their position of a given string.

# In[54]:


text="RegularExpression1IsAn2ImportantTopic3InPython"
pattern=re.compile(r'\d+')

    position = a.start()
    print(number)
    print(position)
        


# Question 22- Write a regular expression in python program to extract maximum/largest numeric value from a string.
# Sample Text:  'My marks in each semester are: 947, 896, 926, 524, 734, 950, 642'
# Expected Output: 950
# 

# In[98]:


Text='My marks in each semester are: 947, 896, 926, 524, 734, 950, 642' 
pattern=re.compile(r'\d+')
marks=[]
matches = re.finditer(pattern,Text)
for a in matches:
    number = a.group()
    marks.append(number)
print(max(marks))


# Question 23- Create a function in python to insert spaces between words starting with capital letters.
# Sample Text: “RegularExpressionIsAnImportantTopicInPython"
# Expected Output: Regular Expression Is An Important Topic In Python
# 

# In[108]:


Text="RegularExpressionIsAnImportantTopicInPython" 
pattern = re.compile(r'([A-Z])')
result = re.sub(pattern, r' \1',Text)
print(result)


# Question 24- Python regex to find sequences of one upper case letter followed by lower case letters

# In[107]:


Text="Regular Expression Is An Important Topic In Python" 
pattern = re.compile(r'\b[A-Z][a-z]+\b')
result = pattern.findall(Text)
print(result)


# Question 25- Write a Python program to remove continuous duplicate words from Sentence using Regular Expression.
# Sample Text: "Hello hello world world"
# Expected Output: Hello hello world
# 

# In[106]:


Text="Hello hello world world"
pattern = re.compile(r'\b(\w+)\s+\1\b')
result = re.sub(pattern, r'\1',Text)
print(result)


# Question 26-  Write a python program using RegEx to accept string ending with alphanumeric character.

# In[117]:


a=input("enter string")
pattern=re.compile(r'\w$')
result=pattern.search(a)
print(bool(result))


# Question 27-Write a python program using RegEx to extract the hashtags.
# Sample Text:  """RT @kapil_kausik: #Doltiwal I mean #xyzabc is "hurt" by #Demonetization as the same has rendered USELESS <ed><U+00A0><U+00BD><ed><U+00B1><U+0089> "acquired funds" No wo"""
# Expected Output: ['#Doltiwal', '#xyzabc', '#Demonetization']
# 

# In[124]:


Text="""RT @kapil_kausik: #Doltiwal I mean #xyzabc is "hurt" by #Demonetization as the same has rendered USELESS <U+00A0><U+00BD><U+00B1><U+0089> "acquired funds" No wo""" 
pattern=re.compile(r'#\w+')
result=pattern.findall(Text)
print(result)


# Question 28- Write a python program using RegEx to remove <U+..> like symbols
# Check the below sample text, there are strange symbols something of the sort <U+..> all over the place. You need to come up with a general Regex expression that will cover all such symbols.
# Sample Text: "@Jags123456 Bharat band on 28??<ed><U+00A0><U+00BD><ed><U+00B8><U+0082>Those who  are protesting #demonetization  are all different party leaders"
# Expected Output: @Jags123456 Bharat band on 28??<ed><ed>Those who  are protesting #demonetization  are all different party leaders
# 
# 

# In[126]:


Text="@Jags123456 Bharat band on 28??<U+00A0><U+00BD><U+00B8><U+0082>Those who are protesting #demonetization are all different party leaders"
pattern=re.sub(r'<U\+[\w]+>', '', Text)
print(pattern)


# Question 29- Write a python program to extract dates from the text stored in the text file.
# Sample Text: Ron was born on 12-09-1992 and he was admitted to school 15-12-1999.
# Note- Store this sample text in the file and then extract dates.
# 

# In[127]:


with open('q29') as file:
    for i in file:
        urls=re.findall(r'\d{1,2}-\d{1,2}-\d{4}',i)
        print(urls)


# Question 30- Create a function in python to remove all words from a string of length between 2 and 4.
# The use of the re.compile() method is mandatory.
# Sample Text: "The following example creates an ArrayList with a capacity of 50 elements. 4 elements are then added to the ArrayList and the ArrayList is trimmed accordingly."
# Expected Output:  following example creates ArrayList a capacity elements. 4 elements added ArrayList ArrayList trimmed accordingly.
# 

# In[137]:


with open("q30") as file:
    for i in file:
        urls=re.sub(r'\b\w{2,4}\b','',i)
        print(urls)


# In[ ]:




