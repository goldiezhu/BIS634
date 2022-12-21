## Relative Severity of COVID-19 by Region in the U.S.

### Data Description
I used two data sets for my final project. The first one is called "Weekly United States COVID-19 Cases and Deaths by State" and is from the CDC Data website. It was last updated by the CDC on Dec. 15, 2022. It has 9120 rows and 10 columns. It's licensing states “Data asset is publicly available to all without restrictions (public).” The second data set is called "Annual Population Estimates, Estimated Components of Resident Population Change, and Rates of the Components of Resident Population Change for the United States, States, District of Columbia, and Puerto Rico: April 1, 2020 to July 1, 2021 (NST-EST2021-ALLDATA)" from the U.S. Census Bureau website. It was very difficult to find an explicit license. There were 57 entries in the population dataset and 31 columns, of which I used two. There was no metadata so the user would have to do summary statistics to find their desired metadata. 

I chose these datasets because they have very precise (unrounded) numbers with consistent updates and could give a clear picture of the pandemic at every point in time. These public data sets can be found on the government websites. I checked for necessary clearning and changes but the data sets were fine in their original state. There was one negative value in the COVID cases data set that I converted to a positive but overall, no missing data. I removed all columns that I would not be using from both data frames. My summary statistics would be skewed by outliers because states with large populations are going to have higher case numbers. 

### Analyses
Because there were so many attributes, I ran a PCA analysis on my dataset to see if there were any clear clusters or groupings of states. There were some clear clusters of color but there may be a lot of overlap too. I standardized the data using SKlearn’s StandardScaler() for my PCA analysis. I then did a K-means scaling for which I standardized the data by dividing the columns by the max value of each column. K-means clustering shows states that have similar rates of deaths and cases so it would offset skew caused by population counts. I was surprised to see that the clustering for k-means was relatively uniform where the majority of the points were.

<img width="659" alt="Screen Shot 2022-12-20 at 11 20 31 AM" src="https://user-images.githubusercontent.com/37753494/209006254-bda7a5ea-1c0e-4a0b-a477-b3b9a93a6a2f.png">
<img width="661" alt="Screen Shot 2022-12-20 at 11 20 24 AM" src="https://user-images.githubusercontent.com/37753494/209006255-0107c48b-9862-45c8-a887-663f4e5fa0c4.png">
<img width="666" alt="Screen Shot 2022-12-20 at 11 20 16 AM" src="https://user-images.githubusercontent.com/37753494/209006258-8b28b98d-fc5e-4e98-84a6-b04786be25d8.png">


### Server
My website runs on Dash, which is for building data visualization interfaces and uses Flask for its backend. This has three components: app.layout, @app.callback, and app.run_server(). The layout uses HTML so you can design your website and function placements. The callback takes inputs and automatically  updates the website when inputs are given. The run_server runs the code and server. I use many dropdown menus, a slider bar, scatterplots, and a chloropleth map.

### Visualizations

This scatter plot visualization tool allows the user to input a 'k' for the k-means clustering as well as choose the x and y axis. The options for the axes are new cases per capita, total cases per capita, new deaths per capita, and total deaths per capita. I was unable to make the continuous color bar on the right into a discrete color display. This is from final_scatter.py.
<img width="1177" alt="Screen Shot 2022-12-20 at 11 23 32 AM" src="https://user-images.githubusercontent.com/37753494/209006303-00f48ae9-3fdc-465e-a505-efb7c38f5bec.png">

This chloropleth allows users to choose the visualization topic, which are the same as the x and y axis choices above. It has a slider bar at the bottom so the user can see the COVID-19 state in the country at different points in time over the pandemic. It says what week it is under the dropdown menu in the photo so the user can see that after they choose an option. This is from the file final_chloropleth.py.
<img width="1186" alt="Screen Shot 2022-12-20 at 11 19 14 AM" src="https://user-images.githubusercontent.com/37753494/209006294-bdcd74fa-4386-4914-a919-df570dbed616.png">

There is a way to create a Dash dashboard but I was unable to so the files have to be run separately for each visualization. 
