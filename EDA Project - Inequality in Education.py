#!/usr/bin/env python
# coding: utf-8

# # EDA PROJECT - EDUCATION INEQUALITY 

# ### INTRODUCTION

# ### In this project, we will conduct exploratory data analysis (EDA) to explore the issue of education inequality on a global scale. The data set is created from Human Development Reports. 

# #### First, import basic libraries for processing data and visualisation 

# In[1]:


import numpy as np 
import pandas as pd               #Processing data
import matplotlib.pyplot as plt   #Visualisation 
import seaborn as sns             #Visualisation


# ### DATA EXPLORATION

# In[2]:


edu = pd.read_csv('Inequality in Education.csv')
edu.head()


# In[3]:


edu.tail()


# In[4]:


#Distribution of countries among different Human Development Groups 
edu['Human Development Groups'].value_counts()


# Most of the countries included in this report belong to high and very high development group (~60%)

# In[5]:


#Distribution of countries among different UNDP Developing Regions
regions = edu['UNDP Developing Regions'].value_counts()
plt.figure(figsize = (6,3))
sns.countplot(x = 'UNDP Developing Regions', data = edu)
plt.title('Number of countries by UNDP Developing Regions')
plt.show()


# The distribution of countries across different UNDP Developing Regions: 
# - Sub-Saharan Africa (SSA): 46 countries 
# - Latin America and the Caribbean (LAC): 33 countries 
# - East Asia and the Pacific (EAP): 26 countries 
# - Arab States (AS): 20 countries 
# - Europe and Central Asia (ECA): 17 countries 
# - South Asia (SA): 9 countries

# Overall, the dataset contains these columns: 
# - ISO3: ISO code for the country/territory 
# - Country: Name of the country/territory 
# - Human Development Groups: Very High, High, Medium, Low 
# - UNDP Developing Regions: SSA, LAC, EAP, AS, ECA, SA
# - HDI Rank (2021): Human Development Index Rank for 2021 
# - Inequality in Education (2010 - 2021): Inequality in education for reported countries from 2010 - 2021

# In[6]:


edu.info()


#  In this dataset, there are 17 columns and 195 entries

# ### Check null & duplicated values

# In[7]:


edu.isnull().sum().plot.bar()
plt.show()


# In[8]:


round(edu.isnull().sum() / 195, 3)


# Approximately 2 - 30% of the data are missing from each column, except for column Country and ISO3

# In[9]:


edu.duplicated().sum()


# There is no duplicated values in the dataset

# ### DATA CLEANING

# Here is some descriptive statistics for the dataset:

# In[10]:


edu.describe(include = 'all')


# Here is the distribution of numerical variables, mostly contain the education inequality scores between 2010 - 2021: 

# In[11]:


edu.hist(figsize = (20,15))
plt.show()


# The histograms show us that the majority of inequality scores in years from 2010 to 2021 are postively skewed. There are more lower inequality scores (<= 20) as compared to higher inequality scores as recorded in the dataset. 

# Let's inspect the data range and any anomalies using boxplots

# In[12]:


edu.boxplot(figsize = (10,6))
plt.xticks(rotation = 45)
plt.show()


# - There seems to be no significant data anomalies in the numerical variables
# - However, both the distribution histograms and boxplots suggest that there is no significant change in inequality scores over time. The question is raised regarding the relevancy/accuracy of the dataset "Is this a good thing given all the world context (e.g: technology developments, increase GDP, etc) in recent years?"

# ### DATA ANALYSIS

# In[13]:


# Average Inequality in Education for each year from 2010 to 2021
mean_inequality_per_year = edu.loc[:, 'Inequality in Education (2010)': 'Inequality in Education (2021)'].mean()

# Plot
plt.figure(figsize=(10,6))
sns.lineplot(x=mean_inequality_per_year.index, y=mean_inequality_per_year.values)
plt.xticks(rotation = 45)
plt.title('Average Inequality in Education (2010-2021)')
plt.xlabel('Year')
plt.ylabel('Average Inequality in Education')
plt.show()


# The line plot shows the average inequality in education for each year from 2010 up to 2021. It shows a graduate decrease in the average inequality in education over that period. This may indiciate a favourable trend for global education for the past decade, even though it still remains a significant issue. 
# 

# #### Some things to note: 
# - The overall inequality does not change over time
# - Inequality in education declined 
# => This means there might be an increase in inequality in other areas that may be worth digging deeper

# In[14]:


#Scatter plot for HDI rank(2021) and Inequality in Education in 2021 
# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='HDI Rank (2021)', y='Inequality in Education (2021)', data=edu, color='green')


# Add a regression line (line of best fit)
sns.regplot(x='HDI Rank (2021)', y='Inequality in Education (2021)', data=edu, scatter=False, color='red')


plt.title('HDI Rank vs Inequality in Education (2021)')
plt.xlabel('HDI Rank (2021)')
plt.ylabel('Inequality in Education (2021)')
plt.show()


# The heat map and scatter plot show the relationship between the Human Development Index (HDI) rank and inequality in education score in 2021 for all countries. 
# As indiciated on the scatter plot, an increase in education inequality is positively associated with an increase in HDI score

# In[15]:


#Change in inequality in education for each country from 2010 to 2021 
edu['Change in Inequality'] = edu['Inequality in Education (2021)'] - edu['Inequality in Education (2010)']  #add new column change


# In[16]:


#Top 10 countries with the highest increase in education inequality
edu.nlargest(10,'Change in Inequality')


# In[17]:


edu.nsmallest(10,'Change in Inequality')


# 
# - The development group is determined by considering multiple factors, which include the inequality. As a result, countries with higher inequality will belong to the lower end groups and vice versa. 
# - Most of the top 10 countries with increasing inequality (70%) are from SSA region. However, the highlight is that Botswana, which is in the same region but got the top improvement in inequality.

# In[18]:


#Analysis by regions

# Group by UNDP Developing Regions and compute the mean for each year
region_mean = edu.groupby('UNDP Developing Regions').mean()

# Select only the Inequality in Education columns
region_mean = region_mean.loc[:, 'Inequality in Education (2010)': 'Inequality in Education (2021)']

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=region_mean.T, dashes=False)
plt.xticks(rotation=45)
plt.title('Average Inequality in Education for Each UNDP Developing Region (2010-2021)')
plt.xlabel('Year')
plt.ylabel('Average Inequality in Education')
plt.legend(title='UNDP Developing Region', labels=region_mean.index)
plt.show()


# Over the years, Europe and Central Asia (ECA) shows lowest average inequality score in education, followed by Latin America and The Caribbean region (LAC)
# Meanwhile, South Asia (SA) and followed by Sub-Saharan Africa (SSA) shows highest average inequality score in education. 
# However, SA and AS show a gradually decreasing trend in average inequality score over the years. Meanwhile, SSA average inequality score in education tends to increase from 2012 as compared to 2010, and remain stable at high score since then till 2021. 

# #### Recommendations:
# 
# - Policymakers may consider cross-regional collaboration and knowledge exchange to share successful strategies for reducing educational inequality.
# - Targeted interventions should address the specific challenges faced by high-inequality regions, focusing on improving access, quality, and inclusivity in education.
# - Monitoring and evaluation systems should be strengthened to assess the impact of policies over time and make data-driven adjustments to educational strategies.
# - Investment in Research:
# Policymakers could invest in research to better understand the contextual factors contributing to educational inequality in each region, enabling the development of more targeted and effective interventions.

# ### IT'S THE END OF THE REPORT. THANKS FOR YOUR ATTENTION. 
