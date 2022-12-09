## Assignment 5
### Goldie Zhu


### Exercise 2:
Describe your plan for your final project. Describe how you want your website to work. (4 points) What do you see as your biggest challenge to completing this, and how do you expect to overcome this challenge? (5 points)

My data is the new U.S. COVID-19 weekly cases dataset that can be found on the CDC data website. I will analyze for changes in total deaths and COVID-19 cases over weeks and months. It'll be interesting because when visualized, you can examine for geographic trends of COVID-19 cases and deaths in different parts of the country. This will be visualized on a full U.S.A. map where the user can choose whether they want to see deaths or cases. The color of each state will correspond to the severity of its COVID-19 situation. They will also be able to choose a time frame from Jan. 2020 to Dec. 2022. In addition, it would be possible to show summary statistics and graphs over the timeframe or for the parameter the user chooses. The visualization could use plotly choropleth and the web app could run on PyWebIO. A challenge I see is that I want to have a slider scale for time inputs so users could instantly see the gradual changes in real time but all the instances I've seen slider used, it must be submitted before the visualization is updated. This may be possible to overcome with a Javascript event listener.
