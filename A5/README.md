## Assignment 5
### Goldie Zhu

### Exercise 1:
This is the second part of Exercise 1 (not the quad-tree/KNN). I upload the dataset, normalize and reduce the data, and plot a PCA scatterplot.

```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.model_selection import train_test_split

data = pd.read_excel('Rice_Cammeo_Osmancik.xlsx', engine='openpyxl')
print(data)

quant = data.drop(['Class'], axis=1)
qual = data['Class']

X_train, X_test, y_train, y_test = train_test_split(quant, qual, train_size=0.8)

#normalize
object = StandardScaler()
std_xtrain = object.fit_transform(X_train)
std_xtest = object.fit_transform(X_test)

#reduce data to two dimensions
pca = decomposition.PCA(n_components=2)

std_xtrain = pd.DataFrame(std_xtrain, columns = X_train.columns)
std_xtest = pd.DataFrame(std_xtest, columns = X_test.columns)

X_train_reduced = pca.fit_transform(std_xtrain)
xtrain_pc0 = X_train_reduced[:, 0]
xtrain_pc1 = X_train_reduced[:, 1]
train_data = []

X_test_reduced = pca.fit_transform(std_xtest)
xtest_pc0 = X_test_reduced[:, 0]
xtest_pc1 = X_test_reduced[:, 1]
test_data = []

train_data = zip(xtrain_pc0, xtrain_pc1, y_train)
test_data = zip(xtest_pc0, xtest_pc1, y_test)

dataDF = pd.DataFrame(train_data)
cammeo = dataDF[dataDF[2] == 'Cammeo']
osmancik = dataDF[dataDF[2] == 'Osmancik']
plt.scatter(cammeo[0], cammeo[1], label='Cammeo')
plt.scatter(osmancik[0], osmancik[1], label='Osmancik')
plt.title('PC0 vs. PC1 for Osmancik and Cammeo Rice')
plt.xlabel('PC0')
plt.ylabel('PC1')
plt.show()
```
<img width="287" alt="Screen Shot 2022-12-09 at 11 02 39 PM" src="https://user-images.githubusercontent.com/37753494/206828074-45c42791-6a0d-40ed-a621-ca0408cd0122.png">

Because the two clusters are visibly separated, KNN should be able to effectively separate and accurately predict the different types of rice.


### Exercise 2:
My data is the new U.S. COVID-19 weekly cases dataset that can be found on the CDC data website. I will analyze for changes in total deaths and COVID-19 cases over weeks and months. It'll be interesting because when visualized, you can examine for geographic trends of COVID-19 cases and deaths in different parts of the country. This will be visualized on a full U.S.A. map where the user can choose whether they want to see deaths or cases. The color of each state will correspond to the severity of its COVID-19 situation. They will also be able to choose a time frame from Jan. 2020 to Dec. 2022. In addition, it would be possible to show summary statistics and graphs over the timeframe or for the parameter the user chooses. The visualization could use plotly choropleth and the web app could run on PyWebIO. It might be one page where you select your chosen criteria and the visualizations appear when the form is submitted. Ideally, the page live updates and and the user could scroll through the different visualizations and statistics. A challenge I see is that I want to have a slider scale for time inputs so users could instantly see the gradual changes in real time but all the instances I've seen slider used, it must be adjusted and submitted before the visualization is updated. This may be possible to overcome with a Javascript event listener.

### Exercise 3:
This CSV dataset is from the National Cancer Institute and its use is regulated by the Public Health Service Act. I removed all lines that did not include relevant data, removed the numbers after each state name, and removed the row representing Nevada because it had no useful data. My Flask wasn't working so I was not able to complete this exercise in time.
