Data Analysis & Visualization of Swiss Glaciers: Predicting Ice Melt.

Swiss glaciers are iconic symbols of the Alps and critical sources of freshwater and the start of some of the 
most important rivers in Europe. Unfortunately, they are rapidly retreating due to climate change. Over the past 
century, glaciers like the Aletsch Glacier and Rhone Glacier have lost significant volume, threatening water 
supplies, ecosystems, and tourism. To measure these impacts, scientists collect various data around the glaciers 
such as temperature and yearly measurements of ice thickness and overall estimated volume to quantify the 
effects of climate change on the glaciers. 

This project aims to analyze historical data on Swiss glaciers and use statistical models to predict glacier 
melt, to understand how scientists collect and use data for their analyzes and predictions.
Visualize trends in glacier retreat.
Considered statistical analyzes include the identification of correlations between climate variables and glacier 
melt, possibly make linear regression of glacier melt as a function of temperature and precipitation.
Use time series to forecast ice melt over the upcoming years. 
Perhaps do some supervised learning for identifying key drivers of glacier retreat.

Sources:
Swiss Glacier Monitoring Network (GLAMOS) for historical glacier data. Gives information on length, mass balance 
and volume variations for the most important swiss glaciers since at least 1973 and for some variables 1950. 
Climate data (temperature, precipitation) from MeteoSwiss or other open source historical databases on climate
and weather.
Functions in python : scikit-learn, statsmodels, pandas, matplotlib, seaborn, plotly

Expected challenges:
I am not an environmental science expert which probably makes the data harder to understand and all the 
possible variables to include in the analyzes more difficult to identify correctly. 
However, one of the advantage of the data is that it is provided by a group of students and scientists at the 
EHZ which, not only makes it more reliable in my opinion, but also allows for the possibility of getting in 
touch with them directly and get more information on the subject.  

Success criteria:
Compare results and forecasts with scientifical reports on the subject. For example, try to predict how many
glaciers will have entirely melted by 2100 and compare the obtained results with that of scientists in published 
reports on the subject. 


Stretch goals:
Implement neural networks (e.g., LSTM) for improved time-series forecasting of glacier melt.

