#!/usr/bin/env python
# coding: utf-8

# Business Case: Netflix - Data Exploration & Visualisation :
# *Business Problem* : Analyze the data and generate insights that could help Netflix in deciding which type of shows/movies to produce and how they can grow the business in different countries

# 1. Defining Problem Statement and Analysing basic metrics

# In[1]:


#Importing Different Libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
     


# In[2]:


import warnings #to ignore the warnings & make our code more representable
warnings.filterwarnings("ignore")


# Loading Of dataset

# In[3]:


df = pd.read_csv("https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/000/940/original/netflix.csv")
df.head()


# "Title" , "director" & "cast" columns needs to be unnested to make our analyis more accurate.
# Duration columns having data in minutes for movies and in seasons for TV shows
# *Attributes information:*
# 
# Show_id: Unique ID for every Movie / Tv Show
# 
# Type: Identifier - A Movie or TV Show
# 
# Title: Title of the Movie / Tv Show
# 
# Director: Director of the Movie
# 
# Cast: Actors involved in the movie/show
# 
# Country: Country where the movie/show was produced
# 
# Date_added: Date it was added on Netflix
# 
# Release_year: Actual Release year of the movie/show
# 
# Rating: TV Rating of the movie/show
# 
# Duration: Total Duration - in minutes or number of seasons
# 
# Listed_in italicized text: Genre
# 
# Description: The summary description

# In[4]:


df


# The dataset contains over 8807 titles, 12 descriptions. Afer a quick view of the data frames, it looks
# like a typical movie/TVshows data frame without raƟngs. We can also see that there are NaN values
# in some columns.

# 2: Observations on the shape of data, data types of all
# the attributes, conversion of categorical attributes to
# 'category' (If required), missing value detection,
# statistical summary

# In[5]:


df.columns #To get All aƩributes


# In[6]:


df.ndim #The shape of data


# In[8]:


df.info() #Data types of all the attributes


# In[ ]:


Statistical Summary Before Data Cleaning:


# In[9]:


df.describe()


# Missing Value Detection
# Data Profiling & Cleaning
# 
# Data Cleaning means the process of idenƟfying incorrect, incomplete, inaccurate, irrelevant, or
# missing pieces of data and then modifying, replacing, or deleting them as needed. Data Cleansing is
# considered as the basic element of Data Science.

# In[10]:


print('\nColumns with missing value:')
print(df.isnull().any())


# In[11]:


df.T.apply(lambda x: x.isnull().sum(), axis = 1)


# From the info, we know that there are 8807 entries and 12 columns to work with for this Exploratory Data Analysis. There
# are a few columns that contain null values, “director,” “cast,” “country,” “date_added,” “rating.”

# In[12]:


df.isnull().sum().sum()


# 4307
# There are a total of 4307 null values across the entire dataset with 2634 missing points under
# "director", 825 under "cast", 831 under "country", 10 under "date_added", 4 under "rating" and
# 3 under “duration ”. We will have to handle all null data points before we can dive into EDA and
# modelling.

# In[ ]:


Imputation is a treatment method for missing value by filling it in using
certain techniques.
Can use mean, mode, or use predicƟve modelling. In thiscase study, we will discuss the use of the
fillna funcƟon from Pandas for this imputaƟon. Drop rows containing missing values. Can use the
dropna funcƟon from Pandas.


# In[13]:


for col in df:
  null_count = df[col].isnull().sum() / len(df) *100
  print(col , "-->" ,null_count)


# In[ ]:


As we can we 30% of Director columns value are missing , we cant drop this much data. We will fill these columns with "Unknown"


# In[14]:


df[["director","cast","country"]] = df[["director","cast","country"]].fillna("Unknown") #Fillling up the missing values


# In[15]:


df.isnull().sum()


# In[ ]:


We will drop these rows in which date added values are missing when we will do the analysis related to date added


# In[16]:


df["rating"].value_counts() #checking unique values in rating columns.


# In[17]:


df.loc[(df["rating"] == "74 min") | (df["rating"] == "84 min") | (df["rating"] == "66 min")]
df["duration"][[5541,5794,5813]] = df["rating"][[5541,5794,5813]]
df["rating"][[5541,5794,5813]] = "Nan"


# In[18]:


df["rating"].value_counts() #checking the count of each category. 
     


# In[19]:


#Conversion of categorical attributes to 'category' and 'datetime'
df["date_added"] = pd.to_datetime(df["date_added"])
df =df.astype({"type" : "category", "rating" : "category"})


# Statistical Summary After Data Cleaning:

# In[20]:


df.describe()


# In[21]:


df.info()


# In[22]:


df.describe()


# In[23]:


df.T.apply(lambda x: x.isnull().sum(), axis = 1)


# *Univariate Analysis*
# Exploratory Analysis and Visualization
# 
# Visual Analysis - Univariate, Bivariate after pre-
# processing of the data
# 
# Univariate analysis
# Analysis done based only on one variable. we are not going to the math
# behind these concepts, for now, let's see what these are in graphs. (please have
# some basic idea on these concepts if you don't get them by seeing graphs).

# In[24]:


df_datetime = df.copy()
df_datetime['Year'] = df.date_added.dt.year  #adding new columns to the dataframe --> year , month , weekday
df_datetime['month'] = df.date_added.dt.month
df_datetime['day'] = df.date_added.dt.day_name()


# In[25]:


sns.countplot(x = "type" , data = df_datetime) #countplot to count the no of movies and tv shows available.
plt.title("No of movies and TV series")
plt.show()


# Immense difference between the count of no of movies and TV show.

# In[26]:


plt.figure(figsize=(20,8))
duration_df = df.loc[df["duration"].str.contains("min")== True]["duration"].apply(lambda x: x.split()[0]).astype(int)  # splting the movies duration as its type is string , extracting the numeri value and converting it into int type
plt.subplot(1,2,1) #subplots to make the data look easy for comparison.
sns.boxplot(duration_df , color = "maroon")
plt.title("Distribution of duration of movies")
duration_seson_df = df.loc[df["duration"].str.contains("Season")== True]["duration"].apply(lambda x: x.split()[0]).astype(int)
plt.subplot(1,2,2)
sns.boxplot(duration_seson_df , color = "maroon")
plt.title("Distribution of no of seasons in TV show")
plt.show()


# In[ ]:


Conclusion -

Average duration of movies are around 100 min
TV shows mostly are having 1 or 2 seasons.
There are lot of outliers present in movies as compare to TV shows


# In[27]:


df_TV_season = df.loc[df["duration"].str.contains("Season")== True , "duration" ].value_counts().reset_index()[:10]  #filtering out top 10 values of TV shows using string. 
df_TV_season.rename(columns = {"index" : "No_of_seasons" , "duration" : "Count"}, inplace = True) #renaming the columns 
plt.figure(figsize=(20,8))
sns.barplot(y = "No_of_seasons" , x = "Count" , data = df_TV_season)
plt.title("Count of TV shows with their no of season")
plt.show()


# In[ ]:


Mostly TV shows have only one season.


# In[ ]:


*Bivariate Analysis*
Bi means two and variate means variable, so here there are two variables. The
analysis is related to cause and the relationship between the two variables. There are
three types of bivariate analysis.


# In[28]:


df_datetime = pd.DataFrame(df)
df_datetime['Year'] = df.date_added.dt.year
df_datetime['month'] = df.date_added.dt.month 
df_datetime['day'] = df.date_added.dt.day_name()
df_datetime_month = df_datetime.sort_values(by ="month")
df_datetime_month['month_name'] = df.date_added.dt.month_name()


# Analysis of number of content added on Netflix over the period

# In[29]:


plt.figure(figsize=(20,8)) #defining fig size fot the graph image
sns.countplot(x = "month_name" , data = df_datetime_month , hue = "type")
plt.title("No of movies and TV series added monthwise") #title name of the plot
plt.legend(loc=(1.01,0.5))
plt.show()


# *Conclusion :-*
# 
# July and December are the months when most content was added becasue no of TV shows durind these two months are maximum among all.
# No of movies added per month is greater then no of TV shows added per month.

# In[30]:


plt.figure(figsize=(20,8))
df_year = df.loc[df['release_year']>2000] #used masked to get out data for movies and TV shows released after 2000
sns.countplot(x='release_year', data = df_year, hue='type')
plt.title("No of movies and TV series added yearwise")
plt.show()
     


# In 2020 , maximum no. of TV shows are added followed by 2019 & 2021.
# More no of movies added on Netflix after "2015"
# We can see in 2021 count of movies add drop significanty ,maybe due to COVID pandemic.

# In[31]:


plt.figure(figsize=(15,8))
sns.countplot(x = "day" , data = df_datetime , hue = "type" ,  order=["Monday" , "Tuesday" , "Wednesday", "Thursday", "Friday", "Saturday" ,"Sunday"])
plt.title("No of movies and TV series added daywise")
plt.show()


# *Conclusion :-* Most of the content added on netflix on "Friday" followed by Thursday as weekend appraches after these days.

# In[32]:


print('PG-13 -----> Parental Guidance with Adult Themes[Parental Guidance]',
'TV-MA -----> Mature Audience[Only for Adults]',
'PG -----> Parental Guidance without Adult Themes[Parental Guidance]',
'TV-14 -----> Contents with Parents strongly cautioned.',
'TV-PG -----> Parental guide suggested[Parental Guidance]',
'TV-Y -----> Children suited content[General Audience & Kids]',
'TV-Y7 -----> Children of age 7 and older[General Audience & Kids]',
'R -----> Strictly for Adults[Only for Adults]',
'TV-G -----> Suitable for all audiences[General Audience & Kids]',
'G -----> General Audience films[General Audience & Kids]',
'NC-17 -----> No one seventeen and under admitted[Only for Adults]',
'NR -----> Not rated movies[Not Rated]',
'TV-Y7-FV -----> Children of age 7 and older with fantasy violence[General Audience & Kids]',
'UR -----> recut version of rated movie[Not Rated]', sep = '\n')

df_rating = df[df["rating"].isnull()== False]
df_rating.reset_index(inplace = True)
plt.figure(figsize=(20,8))
sns.countplot(x ="rating" , data = df_rating , hue = "type")
plt.show()
     


# Conclusion :-
# Mostly TV shows and movies are belongs to TV-MA & TV-14 rating.
# Mostly content available on netflix is for adults and teenagers.

# In[38]:


plt.figure(figsize=(14,6))
movies_ratingwise = df.loc[df["type"] == "Movie" , ["type" , "rating"]]
sns.countplot( y="rating" , data =movies_ratingwise,  palette="Blues_d" )
plt.title("Movies distribution rating wise")
plt.show()


# Conclusion : Mostly movies are belongs to TV-MA & TV-14 rating.

# In[39]:


plt.figure(figsize=(14,6))
movies_ratingwise = df.loc[df["type"] == "TV Show" , ["type" , "rating"]]
sns.countplot( y="rating" , data =movies_ratingwise,  palette="Blues_d" )
plt.title("TV Shows distribution rating wise")
plt.show()


# In[ ]:


Conclusion :- Mostly TV Shows are belongs to TV-MA & TV-14 rating.


# In[40]:


director = df["director"].apply(lambda x : str(x).split(", ")).tolist()  #exploding the nested data in directors column.
df_director = pd.DataFrame(director, index = df["title"])
df_director= df_director.stack()
df_director = df_director.reset_index()
df_director.drop(columns ="level_1" , inplace = True) #droping the columns 
df_director.columns = ["title" , "director"] #renaming the columns
df_fav_director = df.merge(df_director , on = "title" ) #merging of the dataframes
df_fav_director.head(4)
     


# In[41]:


#exploding country column
country = df["country"].apply(lambda x: str(x).split(", ")).tolist() #exploding the country column
df_country = pd.DataFrame(country, index = df["title"])
df_country = df_country.stack()
df_country = df_country.reset_index()
df_country.drop(columns = "level_1" , inplace = True)
df_country.columns = ["title" , "country"]
     

Country_wise_trend = df.merge(df_country , on = "title") #making new dataframe by merfing df_country and original dataframe.
Country_wise_trend.drop(columns = "country_x" , inplace = True)
Country_wise_trend.rename(columns = {"country_y" : "country"}, inplace = True)
Country_wise_trend = Country_wise_trend.loc[Country_wise_trend["country"] != "Unknown"]
top10_country = Country_wise_trend["country"].value_counts().head(10).reset_index()
top10_country.rename(columns = {"index" :"country" , "country" : "count"}, inplace = True)
Country_wise_trend = Country_wise_trend.merge(top10_country, how = "inner" , on = "country")
plt.figure(figsize = (15,8))
sns.countplot(x ="country" , data =Country_wise_trend , hue = "type" )
plt.title("Count of movies and TV shows countrywise")
plt.show()


# Conclusion :-
# Netflix should target to add more TV shows in Unites states and India as compare to Movies.
# Netflix should target to add more Movies in Japan and South Korea.
# 

# In[42]:


#exploding listed_in column
listed_in = df["listed_in"].apply(lambda x: str(x).split(", ")).tolist()
df_genre = pd.DataFrame(listed_in, index = df["title"])
df_genre = df_genre.stack()
df_genre = df_genre.reset_index()
df_genre.drop(columns = "level_1" , inplace = True)
df_genre.columns = ["title" , "genre"]
df_genre.head()
     


# In[54]:


plt.figure(figsize = (18,10))
sns.countplot(y = "genre" , data =df_genre )
plt.title("Ditribution of content Rating_wise")
plt.show()


# Most appearing category in netflix movies and TV shows are:-
# 
# International Movies
# Dramas
# Comedies
# International TV show

# In[ ]:


*Non-Graphical Analysis*


# In[44]:


director_countrywise= df_fav_director.merge(df_country , on = "title")
director_countrywise= director_countrywise.drop(columns = ["director_x" , "country_x" ])
director_countrywise.rename(columns = {"director_y": "director" , "country_y" : "country"}, inplace = True)
director_countrywise = director_countrywise.loc[director_countrywise["director"] != "Unknown"]
director_countrywise.reset_index(inplace= True)
director_countrywise.head()


# In[45]:


country = director_countrywise['country'].value_counts()[:6].index.tolist()
print(' Top 2 Directors of Top 5 Countries')
print('\n')
for val in country:
  if val != 'Unknown':
    print(f'**{val}**')
    print(director_countrywise.loc[director_countrywise['country']==val, 'director'].value_counts()[:2])
    print('\n')


# In[ ]:


Conclusion :

Anurag Kashyap and David Dhawan are the most famous directors for India.
Jay Karas and Marcus Raboyare the most famous directors in United States.


# In[46]:


director_countrywise["director"].value_counts().head(3)
     


# Conclusion : "Rajiv Chilaka" is the most famous director among all followed by Jan Suter

# In[47]:


#exploding cast column
cast = df["cast"].apply(lambda x : str(x).split(", ")).tolist()
df_cast = pd.DataFrame(cast,  index = df["title"])
df_cast = df_cast.stack()
df_cast = df_cast.reset_index()
df_cast.drop(columns = "level_1" , inplace = True)
df_cast.columns = ["title" , "cast"]
df_fav_cast = df.merge(df_cast , on = "title" )
     

cast_countrywise= df_fav_cast.merge(df_country , on = "title")
cast_countrywise= cast_countrywise.drop(columns = ["cast_x" , "country_x"])
cast_countrywise = cast_countrywise.rename(columns = {"cast_y" : "cast" , "country_y" : "country"})
cast_countrywise = cast_countrywise.loc[cast_countrywise["cast"] != "Unknown"].reset_index() #making new dataframe by dropping all rows whose cast is unknown and then resetting the index..00
cast_countrywise.head()


# In[48]:


country_actor = cast_countrywise['country'].value_counts()[:6].index.tolist()
print(' Top 2 Actors of Top 5 Countries')
print('\n')
for val in country:
  if val != 'Unknown':
    print(f'--{val}--')
    print(cast_countrywise.loc[cast_countrywise['country']==val, 'cast'].value_counts()[:2])
    print('\n')


# Conclusion :-
# 
# These are the top two cast of these countires.
# Netflix has added more content for India in which cast are- Anupam Kher or Shah Rukh Khan.

# In[49]:


cast_countrywise["cast"].value_counts().head(5) #value_counts of the cast columns to get the most famous actors


# In[51]:


genre_country_df= df_trend_country.merge(df_genre , on= "title")
genre_country_df.head(5)


# In[52]:


temp_genre = genre_country_df['genre'].value_counts()[:10].reset_index()
temp_genre.rename(columns = {'index':'genre', 'genre':'count'}, inplace=True)
genre_list = temp_genre['genre'].tolist()
df_top10_genre = genre_country_df.loc[genre_country_df['genre'].isin(genre_list)]
df_top10_genre.head()
     


# In[53]:


df_top10_genre = df_top10_genre.loc[df_top10_genre["country"] != "Unknown"]
df_top10_genre["country"].value_counts()[:10]

temp_c = df_top10_genre["country"].value_counts()[:10].reset_index()
temp_c.rename(columns = {'index':'country', 'country':'count'}, inplace=True)
country_list = temp_c["country"].tolist()
df_top10_genre_countrywise = df_top10_genre.loc[df_top10_genre['country'].isin(country_list)]
df_top10_genre_countrywise.head()

heat_genre= pd.DataFrame(df_top10_genre_countrywise.groupby("genre")["country"].value_counts())
heat_genre.rename(columns = {"country" : "count"}, inplace = True)
heat_genre.reset_index(inplace = True)
heat_genre_final = heat_genre.pivot("genre" , "country" , "count")
plt.figure(figsize = (12,8))
sns.heatmap(heat_genre_final , annot = True,  cmap="Blues", fmt = "d")
plt.title("Top 10 genre of 10 differnt countries")
plt.show()


# Conclusion :-
# 
# For India, netflix should add more content of genre International movies , Comedies and Dramas.
# For United States , Netflix should add more content of genre Dramas and Comedy.
# For Canada, Netflix should add more content of genre Dramas & Children and family movies.

# Business Insights :
# 
# With the help of this article, we have been able to learn about-
# 1. Quantity: Our analysis revealed that Netflix had added more movies than TV shows,
# 
# aligning with the expectation that movies dominate their content library.
# 2. Content Addition: July emerged as the month when Netflix adds the most content,
# closely followed by December, indicating a strategic approach to content release.
# 3. Genre Correlation: Strong positive associations were observed between various
# genres, such as TV dramas and international TV shows, romantic and international
# TV shows, and independent movies and dramas. These correlations provide insights
# into viewer preferences and content interconnections.
# 4. Movie Lengths: The analysis of movie durations indicated a peak around the 1960s,
# followed by a stabilization around 100 minutes, highlighting a trend in movie lengths
# over time.
# 5. TV Show Episodes: Most TV shows on Netflix have one season, suggesting a
# preference for shorter series among viewers.
# 
# 6. Common Themes: Words like love, life, family, and adventure were frequently found
# in titles and descriptions, capturing recurring themes in Netflix content.
# 7. Rating Distribution: The distribution of ratings over the years offers insights into the
# evolving content landscape and audience reception.
# 8. Data-Driven Insights: Our data analysis journey showcased the power of data in
# unravelling the mysteries of Netflix’s content landscape, providing valuable insights
# for viewers and content creators.
# 9. Continued Relevance: As the streaming industry evolves, understanding these
# patterns and trends becomes increasingly essential for navigating the dynamic
# landscape of Netflix and its vast library.
# 10. Happy Streaming: We hope this blog has been an enlightening and entertaining
# journey into the world of Netflix, and we encourage you to explore the captivating
# stories within its ever-changing content

# In[ ]:


Recomendations --

1)Netflix should expand its content library to include a broader range of TV shows to cater to the diverse preferences of its audience.

2)Collaborating with renowned directors could lead to the creation of more captivating movies and TV shows, thereby enhancing Netflix's popularity.

3)Additionally, Netflix should consider working with emerging directors who exhibit potential through their limited but highly-rated filmography, potentially due to budget constraints.

4)Exploring different film genres such as horror and comedy, beyond international films, would diversify the content selection.

5)Within the realm of TV shows, focusing on the thriller genre could be beneficial for generating multiple seasons.

6)Increasing the yearly production of original content can demonstrate Netflix's commitment to subscribers, potentially encouraging more people to subscribe.

7)Strategic release timing during holidays, year-end, and weekends can maximize viewership and subscription growth.

8)Some well-received movies could be directly released on Netflix, boosting subscriptions.

9)Leveraging actors with large fan followings for TV shows or web series can attract dedicated viewership.

10)Targeted advertising in countries with limited local content can help Netflix appeal to a broader international audience by showcasing native TV shows.


# In[ ]:




