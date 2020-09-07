#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Import the Dataset. My data has not header and I specify here that header=None
data = pd.read_csv(r"C:\Users\an.makharadze\Desktop\Market_Basket_Optimisation.csv", low_memory=False, header=None)
#Let's see how many rows and columns we have in our dataset
data.shape


# In[3]:


#Let's print top n rows from our dataset
data.head(3)


# In[4]:


#In order to use APRIORI Algorithm we need to install it's library called - apyori using pip install command. Scikit-learn library doesn't include APRIORI algorithm
get_ipython().system('pip install apyori')


# The input data for APRIORI Algorithm should be the list, not the pd.dataframe. So we need to create the list from our dataset.
# At first we need to create an empty list and then append it with the elements in our dataset using 'for' cycle. All of the elements should be strings, so we also need to convert elements into string using command - str

# In[13]:


#Let's create an empty list here
list_of_transactions = []
print (list_of_transactions)


# In[6]:


#Let's append the list with elements converted into string. As we see using shape function above, we have 7501 rows and 20 columns
#So variable i should start from 0 and go to 7501. In each row we need to look at 20 columns. Thats why we are using here the second cycle inside
for i in range(0, 7501):
    list_of_transactions.append([str(data.values[i,j]) for j in range(0, 20)])


# In[7]:


#Let's see our list of transactions
list_of_transactions


# Now it's time to train Apriori on the whole dataset. We do not need here to split dataset into train and test set. 
# At first, we should import apriori algorithm from the library called apyori which we already have installed above
# Then we should define values of the following parameters: min_support, min_confidence, min_lift, min_lenght and max_length according to our business requirements
# If you do not remember the meaning of support, confidence or lift go back and read it again

# In[ ]:


# Training apiori algorithm on our list_of_transactions
from apyori import apriori
rules = apriori(list_of_transactions, min_support = 0.004, min_confidence = 0.2, min_lift = 3, min_length = 2)
#So we will train apriori algorithm on our list_of_transactions and get the rules where items appear together minimum 0.004 times in total transaction (support) and there is minimum 20% chance that item will be added on the cart after purchasing base item (confidence), minimum lift is 3 and minimum number of items in rule is 2


# In[9]:


# Let's create a list of rules and print the results
results = list(rules)
results


# Let's discuss the first rule is -> {'chicken', 'light cream'} with support=0.0045, confidence=0.291 and lift=4.84.
# Please pay attention to that: items_base is {'light cream'} and items_add is {'chicken'}. This means that there is 29% chance (confidence) that user will buy chicken if he has already bought light cream. So left hand side is light cream and right hand side is chicken.

# In[11]:


#In order to visualize our rules better we need to extract elements from our results list, convert it to pd.data frame and sort strong rules by lift value.
#Here is the code for this. We have extracted left hand side and right hand side items from our rules above, also their support, confidence and lift value
def inspect(results):
    lhs     =  [tuple(result [2] [0] [0]) [0] for result in results]
    rhs     =  [tuple(result [2] [0] [1]) [0] for result in results]
    supports = [result [1] for result in results]
    confidences = [result [2] [0] [2]   for result in results]
    lifts = [result [2] [0] [3]   for result in results]
    return list(zip(lhs,rhs,supports,confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results),columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'] )
resultsinDataFrame.head()
    


# In[12]:


#As we have our rules in pd.dataframe we can sort it by lift value using nlargest command. Here we are saying that we need top 12 rule by lift value
resultsinDataFrame.nlargest(n=12, columns='Lift')

