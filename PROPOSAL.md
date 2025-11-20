Visualization and Analysis of Swiss Glaciers data.

Swiss glaciers are iconic symbols of the Alps, critical sources of freshwater and the start of some of the most 
important rivers in Europe. Unfortunately, they are rapidly retreating due to climate change. Glaciers of the Aletsch 
or Rhone for instance have lost significant volume, threatening water supplies and ecosystems. 
To measure these impacts, scientists collect various data around the glaciers such as volume, length or mass balance.
These are then often put into perspective with temperature averages and anomalies to directly quantify the effects of 
climate change on glacier retreat. 


This project aims to analyze historical data on Swiss glaciers and use weather data to implement statistical models that can predict mass balance.

I will start my project with some data visualization that will allow me to better understand it and potentially 
identify some trends that would be useful for the prediction model.

To train and test the data I will use a temporal split approach that allows to check how well the models predict 
future unknown values instead of a spatial split that might be very efficient for predicting present data on unseen 
glaciers based on historical values, but would perhaps not include all the relevant factors that influence changes 
over time and would therefore not be applicable to the future.


Sources: 
Swiss Glacier Monitoring Network (GLAMOS) provides yearly measurements of length change, mass balance 
and volume change for the most important swiss glaciers and over several decades.

Meteoswiss open data for monthly observations of temperature, sunshine and precipitation.

Functions in python : pandas, statsmodels, matplotlib, seaborn, scikit-learn.

Expected challenges:
I am not an environmental science expert which probably makes the data harder to understand and all the 
possible variables to include in the analyzes more difficult to identify correctly. 
However, one of the advantage of the data is that it is provided by scientists at the ETH Zuich which, not only 
makes it more reliable in my opinion, but also allows for the possibility of getting 
more information if needed by directly contacting them. 

Success criteria:
To assess how good the models fit the data I consider using the R² and then to analyze the final results, use the 
RMSE.
Compare results and forecasts with scientifical reports on the subject. 

