#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy import stats
import math
import statistics
import os
import sys
marks = np.asarray([6,7,5,7,7,8,7,6,9,7,4,10,6,8,8,9,5,6,4,8])
def stats(marks):
    print(f"The mean of the marks is {np.mean(marks)}")
    print(f"The median of the marks is {np.median(marks)}")
    print(f"The mode of the marks is {statistics.mode(marks)}")
    print(f"The standard Deviation of the marks is {np.std(marks)}")
    
stats(marks)


# In[3]:


call_records =np.asarray([28, 122, 217, 130, 120, 86, 80, 90, 140, 120, 70, 40, 145, 113, 90, 68, 174, 194, 170,100, 75, 104, 97, 75,123, 100, 75, 104, 97, 75, 123, 100, 89, 120, 109])
stats(call_records)


# In[4]:


x = np.asarray([0,1,2,3,4,5])
f_x = np.array([0.09,0.15,0.40,0.25,0.10,0.01]) 
x.reshape((1,-1))
f_x.reshape((-1,1))
mean=np.dot(x,f_x)
variance_of_x=(x-mean)**2
variance = np.dot(variance_of_x.reshape(1,-1),f_x)
print(f"Mean no. of workouts: {mean}")
print(f"Variance of workouts: {variance}")


# In[5]:


x = np.asarray([0,1,2,3,4,5])
f_x = np.array([0.09,0.15,0.40,0.25,0.10,0.01]) 
x.reshape((1,-1))
f_x.reshape((-1,1))
mean=np.dot(x,f_x)
variance_of_x=(x-mean)**2
variance = np.dot(variance_of_x.reshape(1,-1),f_x)
print(f"Mean no. of workouts: {mean}")
print(f"Variance of workouts: {variance}")


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import scipy.special
#x = faulty = 0.3
#y = not faulty = 0.7
x = 0.3
y = 0.7
df=pd.DataFrame({'a':[int(i) for i in range(7)],
                 'B_a':[scipy.special.comb(6,i)*(x**i)*(y**(6-i)) for i in range(7)]})
print(df.iloc[2])
plt.figure(figsize=(15,5))
sns.barplot('a','B_a',data=df)
plt.xlabel('Number of Faulty Leds')
plt.ylabel('Probability of Fault')


# In[7]:


df['Expected value']=df['a']*df['B_a']
mean=np.round(df['Expected value'].sum())
print('mean = {}'.format(mean))
df['variance']=df['B_a']*(df['a']-mean)**2
std=np.sqrt(df['variance'].sum())
print(f"Standard Deviation : {np.round(std)}")


# In[8]:


from scipy.stats import binom
import numpy as np

print(f"Probability of each of them solving 5 questions correctly is:{binom.pmf(5,8,0.75)*binom.pmf(5,12,0.45)}")
print(f"Probability of each of them solving 4,6 questions correctly is:{binom.pmf(4,8,0.75)*binom.pmf(6,12,0.45)}")
def binom_plot(n,p,):
    fig,ax=plt.subplots(1,1)
    x = np.arange(binom.ppf(0.01, n, p),binom.ppf(0.99, n, p))
    ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')
    ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)


# In[13]:


from scipy.stats import poisson
#We need to calculate average number of customers arriving per 4 minutes
#72/60 customers come per minute
mu = 4*(72/60) #customers come per 4 minutes
print(f"The probability of arriving 5 cutomers in 4 minutes is : {poisson.pmf(k=5,mu=mu)}")
print(f"The probability of arriving not more than 3 customers in 4 minutes is : {poisson.pmf(k=3, mu=mu)}")
print(f'The Probability of more than 3 customers arriving in 4 minutes is : {1-poisson.cdf(k=3,mu=mu)}')
The probability of arriving 5 cutomers in 4 minutes is : 0.17474768364388296
The probability of arriving not more than 3 customers in 4 minutes is : 0.15169069760753714
The Probability of more than 3 customers arriving in 4 minutes is : 0.7057700835034357
x = list(range(0,10))
fig,ax = plt.subplots(1,1,figsize=(15,5))
ax.plot(x, poisson.pmf(x,mu), 'bo', ms=8, label='poisson pmf')
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
plt.xlabel('Number of customers')
plt.ylabel('Probability')


# In[14]:


from scipy.stats import poisson
#Rate of entering=77 per minute
#error rate= 6/hour=0.1 per minute
#No of errors per word=0.1/77
unit_mu=0.1/77
def mu(n):
    return n * unit_mu
print(f"The pobability of commiting 2 errors in 455 words financial report is :{poisson.pmf(2,mu=mu(455))}")
print(f"The pobability of commiting 2 errors in 1000 words financial report is :{poisson.pmf(2,mu=mu(1000))}")
print(f"The pobability of commiting 2 errors in 255 words financial report is :{poisson.pmf(2,mu=mu(255))}")
x=range(100,1000,50)
mu=[i*unit_mu for i in x]
fig,ax = plt.subplots(1,1,figsize=(15,5))
ax.plot(x,poisson.pmf(2,mu), 'bo', ms=8, label='poisson pmf')
ax.vlines(x,0, poisson.pmf(2,mu), colors='b', lw=5, alpha=0.5)


# In[26]:


from scipy.stats import norm
from scipy.integrate import quad
def P(z,b=-np.inf) :
    return integrate.quad(norm.pdf,b,z)[0]

print('P(Z>1.26) = %.5f'%(1-P(1.26)))
print('P(Z<-0.86) = %.5f'%P(-0.86))
print('P(Z>-1.37) = %.5f'%(1-P(-1.37)))
print('P(−1.25 < Z < 0.37) = %.5f'%P(0.37,b=-1.25))
print('P(Z ≤ −4.6) = %.5f'%P(-4.6))


# In[27]:


mean = 10
std = np.sqrt(4)
import scipy.integrate
def I(z, b=-np.inf):
    z = (z-mean)/std
    return integrate.quad(norm.pdf,b,z)[0]
print(f"Probability that current > 13mA is: {1-I(13)}")
print(f"Probability that current is between 9 mA and 11 mA is : {1-I(11,b=9)}")


# In[28]:


mean_dia=0.2508
std_dia=0.0005
#specified dia in the range of 0.2485<d<0.2515
#case-1 if mean_dia=0.2508
def I(mean,std,a,b) :
  #gives P(Z<=x)
  a=(a-mean)/std
  b=(b-mean)/std
print(f"Proportion of shafts with dia in range of 0.2485<d<0.2515 when mean diameter:{0.2508,I(0.2508,0.0005,0.2485,0.2515)}")
print(f"Proportion of shafts with dia in range of 0.2485<d<0.2515 when mean diameter:{0.2500,I(0.2500,0.0005,0.2485,0.2515)}")


# In[ ]:





# In[ ]:




