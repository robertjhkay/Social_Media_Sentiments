```python
# Libraries I will be using as part of the analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

The dataset, titled 'Social Media Sentiments Analysis Dataset,' offers a rich tapestry of human emotions, trends, and interactions as observed across diverse social media platforms. It serves as a snapshot of user-generated content, encapsulating a myriad of elements including textual narratives, timestamps, hashtags, geographic locations, likes, and retweets. By extracting this dataset from www.kaggle.com, we gain access to a wealth of information that sets the stage for comprehensive analysis and insights into the dynamics of social media discourse.


```python
def extract_from_csv(address):
    df = pd.read_csv(address, encoding = 'ISO-8859-1')
    return df

social_media_df = extract_from_csv('sentimentdataset.csv')

# Displaying the variables' names and the first few entries to inform further analysis
social_media_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>Text</th>
      <th>Sentiment</th>
      <th>Timestamp</th>
      <th>User</th>
      <th>Platform</th>
      <th>Hashtags</th>
      <th>Retweets</th>
      <th>Likes</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>Enjoying a beautiful day at the park!        ...</td>
      <td>Positive</td>
      <td>2023-01-15 12:30:00</td>
      <td>User123</td>
      <td>Twitter</td>
      <td>#Nature #Park</td>
      <td>15.0</td>
      <td>30.0</td>
      <td>USA</td>
      <td>2023</td>
      <td>1</td>
      <td>15</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Traffic was terrible this morning.           ...</td>
      <td>Negative</td>
      <td>2023-01-15 08:45:00</td>
      <td>CommuterX</td>
      <td>Twitter</td>
      <td>#Traffic #Morning</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>Canada</td>
      <td>2023</td>
      <td>1</td>
      <td>15</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>Just finished an amazing workout! ðª       ...</td>
      <td>Positive</td>
      <td>2023-01-15 15:45:00</td>
      <td>FitnessFan</td>
      <td>Instagram</td>
      <td>#Fitness #Workout</td>
      <td>20.0</td>
      <td>40.0</td>
      <td>USA</td>
      <td>2023</td>
      <td>1</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>Excited about the upcoming weekend getaway!  ...</td>
      <td>Positive</td>
      <td>2023-01-15 18:20:00</td>
      <td>AdventureX</td>
      <td>Facebook</td>
      <td>#Travel #Adventure</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>UK</td>
      <td>2023</td>
      <td>1</td>
      <td>15</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>Trying out a new recipe for dinner tonight.  ...</td>
      <td>Neutral</td>
      <td>2023-01-15 19:55:00</td>
      <td>ChefCook</td>
      <td>Instagram</td>
      <td>#Cooking #Food</td>
      <td>12.0</td>
      <td>25.0</td>
      <td>Australia</td>
      <td>2023</td>
      <td>1</td>
      <td>15</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python

print(len(social_media_df))
```

    732
    

I can see there is a lot of text within the data set. The only numerical data is 'Likes' and 'Retweets'. Looking at the top twenty in each category seems a logical place to start.


```python
top_twenty_likes = social_media_df.nlargest(20, 'Likes')
print(top_twenty_likes)
```

         Unnamed: 0.1  Unnamed: 0  \
    335           336         340   
    345           346         350   
    355           356         360   
    368           369         373   
    382           383         387   
    402           403         407   
    432           433         437   
    470           471         475   
    481           482         486   
    510           511         515   
    520           521         525   
    530           531         535   
    540           541         545   
    550           551         555   
    560           561         565   
    570           571         575   
    339           340         344   
    347           348         352   
    356           357         361   
    364           365         369   
    
                                                      Text         Sentiment  \
    335  Thrilled to witness the grandeur of a cultural...    Thrill           
    345  Motivated to achieve fitness goals after an in...    Motivation       
    355  Anticipation for an upcoming adventure in an e...    Anticipation     
    368  Elation over discovering a rare book in a quai...    Elation          
    382  A sense of wonder at the vastness of the cosmo...       Wonder        
    402  Awe-inspired by the vastness of the cosmos on ...     Wonder          
    432  Heartache deepens, a solitary journey through ...        Despair      
    470  Dancing on sunshine, each step a celebration o...              Joy    
    481  Surrounded by the colors of joy, a canvas pain...              Joy    
    510  At the front row of Adele's concert, each note...          Emotion    
    520  At a Justin Bieber concert, the infectious bea...       Enthusiasm    
    530  Captivated by the spellbinding plot twists, th...       Excitement    
    540  Celebrating a historic victory in the World Cu...              Joy    
    550  After a series of defeats, the soccer team fac...   Disappointment    
    560  In the serene beauty of a sunset, nature unfol...      Tranquility    
    570  Underneath the city lights, the dancer express...      Mesmerizing    
    339  Reflecting on life's journey, grateful for the...      Gratitude      
    347  Feeling empowered after conquering a challengi...    Empowerment      
    356  Reflecting on personal growth achieved through...    Reflection       
    364  Wonderment at the beauty of a double rainbow a...    Wonderment       
    
                   Timestamp                                   User     Platform  \
    335  2020-09-15 14:45:00                     CultureEnthusiast    Instagram    
    345  2022-02-28 07:15:00                       FitnessJunkie       Facebook    
    355  2022-07-25 10:00:00                       Wanderlust          Twitter     
    368  2018-09-22 16:30:00                       Bookworm           Instagram    
    382  2018-07-08 23:00:00                       CosmosExplorer     Instagram    
    402  2018-07-08 23:00:00                       CosmosExplorer     Instagram    
    432  2022-08-18 22:00:00                       SolitaryDescent    Instagram    
    470  2021-08-20 15:45:00                        SunshineDancer    Instagram    
    481  2019-07-02 17:00:00                      ColorfulLaughter    Instagram    
    510  2022-09-15 20:00:00                      AdeleConcertGoer    Instagram    
    520  2018-08-05 20:00:00                 BieberDanceEnthusiast    Instagram    
    530  2018-10-12 20:00:00       MovieEnthusiastPremiereAttendee      Twitter    
    540  2018-07-15 21:30:00        FootballFanWorldCupCelebration    Instagram    
    550  2019-11-02 18:45:00                  SoccerFanTeamDefeats      Twitter    
    560  2021-07-10 19:00:00          NatureEnthusiastSunsetWonder    Instagram    
    570  2018-06-15 22:00:00   DanceEnthusiastCityNightPerformance      Twitter    
    339  2018-12-05 08:45:00                         LifeLearner       Twitter     
    347  2018-09-05 14:20:00                       HikingExplorer     Instagram    
    356  2019-03-08 18:30:00                       GrowthSeeker       Instagram    
    364  2017-06-12 18:20:00                       RainbowChaser       Twitter     
    
                                             Hashtags  Retweets  Likes  \
    335    #Thrill #CulturalCelebration                    40.0   80.0   
    345    #Motivation #FitnessGoals                       40.0   80.0   
    355     #Anticipation #AdventureAwaits                 40.0   80.0   
    368     #Elation #RareBookDiscovery                    40.0   80.0   
    382      #Wonder #StargazingAdventure                  40.0   80.0   
    402      #Wonder #StargazingAdventure                  40.0   80.0   
    432       #Despair #AbyssOfHeartache                   40.0   80.0   
    470                          #Joy #SimpleMoments       40.0   80.0   
    481                          #Joy #EndlessSmiles       40.0   80.0   
    510                       #Emotion #AdeleConcert       40.0   80.0   
    520                    #Enthusiasm #JustinBieber       40.0   80.0   
    530            #Excitement #MoviePremiereThrills       40.0   80.0   
    540                        #Joy #WorldCupTriumph       40.0   80.0   
    550               #Disappointment #SoccerDefeats       40.0   80.0   
    560                   #Tranquility #SunsetBeauty       40.0   80.0   
    570          #Mesmerizing #NightDancePerformance       40.0   80.0   
    339   #Gratitude #LifeLessons                          35.0   70.0   
    347    #Empowerment #HikingAdventure                   35.0   70.0   
    356     #Reflection #PersonalGrowth                    35.0   70.0   
    364     #Wonderment #DoubleRainbow                     35.0   70.0   
    
                      Country  Year  Month  Day  Hour  
    335   India                2020      9   15    14  
    345   Australia            2022      2   28     7  
    355    India               2022      7   25    10  
    368    USA                 2018      9   22    16  
    382      South Africa      2018      7    8    23  
    402      South Africa      2018      7    8    23  
    432         South Africa   2022      8   18    22  
    470                  USA   2021      8   20    15  
    481               Canada   2019      7    2    17  
    510                  USA   2022      9   15    20  
    520               Canada   2018      8    5    20  
    530                  USA   2018     10   12    20  
    540               Brazil   2018      7   15    21  
    550               Brazil   2019     11    2    18  
    560               Canada   2021      7   10    19  
    570                  USA   2018      6   15    22  
    339    Brazil              2018     12    5     8  
    347    USA                 2018      9    5    14  
    356    USA                 2019      3    8    18  
    364    USA                 2017      6   12    18  
    


```python
top_twenty_retweets = social_media_df.nlargest(20, 'Retweets')
print(top_twenty_retweets)
```

         Unnamed: 0.1  Unnamed: 0  \
    335           336         340   
    345           346         350   
    355           356         360   
    368           369         373   
    382           383         387   
    402           403         407   
    432           433         437   
    470           471         475   
    481           482         486   
    510           511         515   
    520           521         525   
    530           531         535   
    540           541         545   
    550           551         555   
    560           561         565   
    570           571         575   
    339           340         344   
    347           348         352   
    356           357         361   
    364           365         369   
    
                                                      Text         Sentiment  \
    335  Thrilled to witness the grandeur of a cultural...    Thrill           
    345  Motivated to achieve fitness goals after an in...    Motivation       
    355  Anticipation for an upcoming adventure in an e...    Anticipation     
    368  Elation over discovering a rare book in a quai...    Elation          
    382  A sense of wonder at the vastness of the cosmo...       Wonder        
    402  Awe-inspired by the vastness of the cosmos on ...     Wonder          
    432  Heartache deepens, a solitary journey through ...        Despair      
    470  Dancing on sunshine, each step a celebration o...              Joy    
    481  Surrounded by the colors of joy, a canvas pain...              Joy    
    510  At the front row of Adele's concert, each note...          Emotion    
    520  At a Justin Bieber concert, the infectious bea...       Enthusiasm    
    530  Captivated by the spellbinding plot twists, th...       Excitement    
    540  Celebrating a historic victory in the World Cu...              Joy    
    550  After a series of defeats, the soccer team fac...   Disappointment    
    560  In the serene beauty of a sunset, nature unfol...      Tranquility    
    570  Underneath the city lights, the dancer express...      Mesmerizing    
    339  Reflecting on life's journey, grateful for the...      Gratitude      
    347  Feeling empowered after conquering a challengi...    Empowerment      
    356  Reflecting on personal growth achieved through...    Reflection       
    364  Wonderment at the beauty of a double rainbow a...    Wonderment       
    
                   Timestamp                                   User     Platform  \
    335  2020-09-15 14:45:00                     CultureEnthusiast    Instagram    
    345  2022-02-28 07:15:00                       FitnessJunkie       Facebook    
    355  2022-07-25 10:00:00                       Wanderlust          Twitter     
    368  2018-09-22 16:30:00                       Bookworm           Instagram    
    382  2018-07-08 23:00:00                       CosmosExplorer     Instagram    
    402  2018-07-08 23:00:00                       CosmosExplorer     Instagram    
    432  2022-08-18 22:00:00                       SolitaryDescent    Instagram    
    470  2021-08-20 15:45:00                        SunshineDancer    Instagram    
    481  2019-07-02 17:00:00                      ColorfulLaughter    Instagram    
    510  2022-09-15 20:00:00                      AdeleConcertGoer    Instagram    
    520  2018-08-05 20:00:00                 BieberDanceEnthusiast    Instagram    
    530  2018-10-12 20:00:00       MovieEnthusiastPremiereAttendee      Twitter    
    540  2018-07-15 21:30:00        FootballFanWorldCupCelebration    Instagram    
    550  2019-11-02 18:45:00                  SoccerFanTeamDefeats      Twitter    
    560  2021-07-10 19:00:00          NatureEnthusiastSunsetWonder    Instagram    
    570  2018-06-15 22:00:00   DanceEnthusiastCityNightPerformance      Twitter    
    339  2018-12-05 08:45:00                         LifeLearner       Twitter     
    347  2018-09-05 14:20:00                       HikingExplorer     Instagram    
    356  2019-03-08 18:30:00                       GrowthSeeker       Instagram    
    364  2017-06-12 18:20:00                       RainbowChaser       Twitter     
    
                                             Hashtags  Retweets  Likes  \
    335    #Thrill #CulturalCelebration                    40.0   80.0   
    345    #Motivation #FitnessGoals                       40.0   80.0   
    355     #Anticipation #AdventureAwaits                 40.0   80.0   
    368     #Elation #RareBookDiscovery                    40.0   80.0   
    382      #Wonder #StargazingAdventure                  40.0   80.0   
    402      #Wonder #StargazingAdventure                  40.0   80.0   
    432       #Despair #AbyssOfHeartache                   40.0   80.0   
    470                          #Joy #SimpleMoments       40.0   80.0   
    481                          #Joy #EndlessSmiles       40.0   80.0   
    510                       #Emotion #AdeleConcert       40.0   80.0   
    520                    #Enthusiasm #JustinBieber       40.0   80.0   
    530            #Excitement #MoviePremiereThrills       40.0   80.0   
    540                        #Joy #WorldCupTriumph       40.0   80.0   
    550               #Disappointment #SoccerDefeats       40.0   80.0   
    560                   #Tranquility #SunsetBeauty       40.0   80.0   
    570          #Mesmerizing #NightDancePerformance       40.0   80.0   
    339   #Gratitude #LifeLessons                          35.0   70.0   
    347    #Empowerment #HikingAdventure                   35.0   70.0   
    356     #Reflection #PersonalGrowth                    35.0   70.0   
    364     #Wonderment #DoubleRainbow                     35.0   70.0   
    
                      Country  Year  Month  Day  Hour  
    335   India                2020      9   15    14  
    345   Australia            2022      2   28     7  
    355    India               2022      7   25    10  
    368    USA                 2018      9   22    16  
    382      South Africa      2018      7    8    23  
    402      South Africa      2018      7    8    23  
    432         South Africa   2022      8   18    22  
    470                  USA   2021      8   20    15  
    481               Canada   2019      7    2    17  
    510                  USA   2022      9   15    20  
    520               Canada   2018      8    5    20  
    530                  USA   2018     10   12    20  
    540               Brazil   2018      7   15    21  
    550               Brazil   2019     11    2    18  
    560               Canada   2021      7   10    19  
    570                  USA   2018      6   15    22  
    339    Brazil              2018     12    5     8  
    347    USA                 2018      9    5    14  
    356    USA                 2019      3    8    18  
    364    USA                 2017      6   12    18  
    


```python
social_media_df[['Retweets','Likes']].corr() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Retweets</th>
      <th>Likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Retweets</th>
      <td>1.000000</td>
      <td>0.998482</td>
    </tr>
    <tr>
      <th>Likes</th>
      <td>0.998482</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.regplot(x="Retweets", y="Likes", data = social_media_df)

# Saving the plot to an image file
plt.savefig('plot1.png')
```


    
![png](Social_Media_Sentiments_files/Social_Media_Sentiments_8_0.png)
    


I can see that the top twenty are the same in both cases. Looking at the Pearson Correlation Coefficient there is a strong linear relationship between the two. Also, there are many different hashtag and sentiment descriptors which makes analysis here difficult. Text variables that are more discrete are 'Platform' and 'Country'.


```python
# Apply strip() function to string columns
social_media_df = social_media_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Replace "Scotland" with "UK" in the 'Country' column
social_media_df['Country'] = social_media_df['Country'].replace('Scotland', 'UK')

# Get unique values in the 'Country' column
unique_countries = social_media_df['Country'].unique()

# Print unique countries
print(unique_countries)
```

    ['USA' 'Canada' 'UK' 'Australia' 'India' 'France' 'Brazil' 'Japan'
     'Greece' 'Germany' 'Sweden' 'Italy' 'Netherlands' 'South Africa' 'Spain'
     'Portugal' 'Switzerland' 'Austria' 'Belgium' 'Denmark' 'Czech Republic'
     'Jordan' 'Peru' 'Maldives' 'China' 'Cambodia' 'Norway' 'Colombia'
     'Ireland' 'Jamaica' 'Kenya' 'Thailand']
    


```python
# Count the occurrences of each country
country_counts = social_media_df['Country'].value_counts()

# Plot the bar plot
country_counts.plot(kind='bar', color='skyblue')

# Set the title and labels
plt.title('Distribution of Countries')
plt.xlabel('Country')
plt.ylabel('Count')

# Saving the plot to an image file
plt.savefig('plot2.png')

# Show the plot
plt.show()
```


    
![png](Social_Media_Sentiments_files/Social_Media_Sentiments_11_0.png)
    


From this we can see that although many countries have entries, the majority (over two-thirds) comes from the top 5. This maybe because the platforms we have extracted data from are prosperous in these countries and not in the others. China, for example, has a very small presence here.


```python
# Count the occurrences of each country
platform_counts = social_media_df['Platform'].value_counts()

# Plot the bar plot
platform_counts.plot(kind='bar', color='skyblue')

# Set the title and labels
plt.title('Distribution of Platform')
plt.xlabel('Platform')
plt.ylabel('Count')

# Saving the plot to an image file
plt.savefig('plot3.png')

# Show the plot
plt.show()
```


    
![png](Social_Media_Sentiments_files/Social_Media_Sentiments_13_0.png)
    



```python
sns.boxplot(x="Platform", y="Likes", data = social_media_df)

# Saving the plot to an image file
plt.savefig('plot4.png')
```


    
![png](Social_Media_Sentiments_files/Social_Media_Sentiments_14_0.png)
    



```python
platform_counts_by_year = social_media_df.groupby(['Platform', 'Year']).size().reset_index(name='Counts')

# Create a line plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=platform_counts_by_year, x='Year', y='Counts', hue='Platform', marker='o')
plt.title('Platform Usage over the Years')
plt.xlabel('Year')
plt.ylabel('Counts')
plt.grid(True)
plt.legend()

# Saving the plot to an image file
plt.savefig('plot5.png')

plt.show()
```


    
![png](Social_Media_Sentiments_files/Social_Media_Sentiments_15_0.png)
    



    <Figure size 640x480 with 0 Axes>


Several key insights from the data analysis:

**Similar Number of Entries Across Platforms**: The distribution of entries across platforms is relatively consistent, indicating a comparable level of activity on each platform.

**Instagram Receives More Likes**: The boxplot reveals that Instagram posts tend to receive more likes compared to other platforms. Specifically, the top 25% of Instagram posts receive between 55 to 80 likes.

**Twitter is Most Popular Over the Years**: Despite Instagram receiving more likes per post, the analysis of platform usage over the years shows that Twitter is currently the most popular platform.

**Similar Profile Across Platforms**: Despite differences in popularity and engagement metrics, all three platforms exhibit a similar profile, suggesting that users may engage with content similarly across different platforms.

These insights provide valuable information about user behavior and platform dynamics, which can inform strategic decisions related to content creation, platform selection, and audience engagement strategies.



```python
# Selecting relevant columns from the DataFrame
df_gp = social_media_df[['Country','Platform','Likes']]

# Grouping the data by 'Country' and 'Platform', and summing the 'Likes'
group_all = df_gp.groupby(['Country','Platform'], as_index=False).sum()

group_all
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Platform</th>
      <th>Likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>Facebook</td>
      <td>1056.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Australia</td>
      <td>Instagram</td>
      <td>1003.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Australia</td>
      <td>Twitter</td>
      <td>867.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Austria</td>
      <td>Facebook</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Belgium</td>
      <td>Twitter</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brazil</td>
      <td>Facebook</td>
      <td>290.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Brazil</td>
      <td>Instagram</td>
      <td>330.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Brazil</td>
      <td>Twitter</td>
      <td>280.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cambodia</td>
      <td>Instagram</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Canada</td>
      <td>Facebook</td>
      <td>1760.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Canada</td>
      <td>Instagram</td>
      <td>1867.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Canada</td>
      <td>Twitter</td>
      <td>1861.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>China</td>
      <td>Twitter</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Colombia</td>
      <td>Facebook</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Czech Republic</td>
      <td>Facebook</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Denmark</td>
      <td>Instagram</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>France</td>
      <td>Facebook</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>France</td>
      <td>Instagram</td>
      <td>250.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>France</td>
      <td>Twitter</td>
      <td>375.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Germany</td>
      <td>Facebook</td>
      <td>370.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Germany</td>
      <td>Instagram</td>
      <td>115.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Germany</td>
      <td>Twitter</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Greece</td>
      <td>Facebook</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Greece</td>
      <td>Twitter</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>India</td>
      <td>Facebook</td>
      <td>746.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>India</td>
      <td>Instagram</td>
      <td>756.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>India</td>
      <td>Twitter</td>
      <td>1173.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Ireland</td>
      <td>Instagram</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Italy</td>
      <td>Facebook</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Italy</td>
      <td>Instagram</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Italy</td>
      <td>Twitter</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Jamaica</td>
      <td>Instagram</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Japan</td>
      <td>Facebook</td>
      <td>335.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Japan</td>
      <td>Instagram</td>
      <td>270.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Japan</td>
      <td>Twitter</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Jordan</td>
      <td>Twitter</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Kenya</td>
      <td>Instagram</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Maldives</td>
      <td>Facebook</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Netherlands</td>
      <td>Facebook</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Netherlands</td>
      <td>Instagram</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Norway</td>
      <td>Instagram</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Peru</td>
      <td>Facebook</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Portugal</td>
      <td>Facebook</td>
      <td>110.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>South Africa</td>
      <td>Instagram</td>
      <td>430.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Spain</td>
      <td>Instagram</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Spain</td>
      <td>Twitter</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Sweden</td>
      <td>Instagram</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Switzerland</td>
      <td>Instagram</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Switzerland</td>
      <td>Twitter</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Thailand</td>
      <td>Instagram</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>UK</td>
      <td>Facebook</td>
      <td>1613.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>UK</td>
      <td>Instagram</td>
      <td>2356.0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>UK</td>
      <td>Twitter</td>
      <td>1903.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>USA</td>
      <td>Facebook</td>
      <td>2517.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>USA</td>
      <td>Instagram</td>
      <td>3381.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>USA</td>
      <td>Twitter</td>
      <td>2460.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# List of countries of interest
countries_of_interest = ['USA', 'UK', 'Canada', 'Australia', 'India']

# Filtering the grouped DataFrame to include only the specified countries
group_all_filtered = group_all[group_all['Country'].isin(countries_of_interest)]

grouped_pivot = group_all_filtered.pivot(index='Country',columns='Platform')
grouped_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Likes</th>
    </tr>
    <tr>
      <th>Platform</th>
      <th>Facebook</th>
      <th>Instagram</th>
      <th>Twitter</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Australia</th>
      <td>1056.0</td>
      <td>1003.0</td>
      <td>867.0</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>1760.0</td>
      <td>1867.0</td>
      <td>1861.0</td>
    </tr>
    <tr>
      <th>India</th>
      <td>746.0</td>
      <td>756.0</td>
      <td>1173.0</td>
    </tr>
    <tr>
      <th>UK</th>
      <td>1613.0</td>
      <td>2356.0</td>
      <td>1903.0</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>2517.0</td>
      <td>3381.0</td>
      <td>2460.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a figure and axis for the heatmap
fig, ax = plt.subplots()

# Plot the heatmap with a diverging colormap
im = ax.pcolor(grouped_pivot, cmap='RdBu_r')

# Get the row and column labels
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# Move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# Set labels for ticks
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

# Rotate x-axis labels if they are too long
plt.xticks(rotation=90)

# Add colorbar
fig.colorbar(im)

# Saving the plot to an image file
plt.savefig('plot6.png')

# Display the plot
plt.show()
```


    
![png](Social_Media_Sentiments_files/Social_Media_Sentiments_19_0.png)
    


***Final Conclusions:***
- The top five countries for using 'Instagram', 'Twitter', and 'Facebook' are the USA, UK, Australia, Canada, and India.
- There is an incredibly strong linear relationship between 'Likes' and 'Retweets'.
- Usage over the years has been very similar across the platforms
- USA favors 'Instagram' whist Canada has no real preference

