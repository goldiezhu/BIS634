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
plt.legend()
plt.show()
```
<img width="287" alt="Screen Shot 2022-12-09 at 11 24 19 PM" src="https://user-images.githubusercontent.com/37753494/206828831-9cbd24c4-ee62-4ec2-9369-77bb30e072e5.png">

Because the two clusters are basically visibly separated, KNN should be able to effectively separate and accurately predict the different types of rice. There might be some issues in the slight overlap area but that area is small compared to the cluster size.

I tried to do the 2D quad-tree/KNN but was not able to make it work. Failed code is in the appendix.


### Exercise 2:
My data is the new U.S. COVID-19 weekly cases dataset that can be found on the CDC data website. I will analyze for changes in total deaths and COVID-19 cases over weeks and months. It'll be interesting because when visualized, you can examine for geographic trends of COVID-19 cases and deaths in different parts of the country. This will be visualized on a full U.S.A. map where the user can choose whether they want to see deaths or cases. The color of each state will correspond to the severity of its COVID-19 situation. They will also be able to choose a time frame from Jan. 2020 to Dec. 2022. In addition, it would be possible to show summary statistics and graphs over the timeframe or for the parameter the user chooses. The visualization could use plotly choropleth and the web app could run on PyWebIO. It might be one page where you select your chosen criteria and the visualizations appear when the form is submitted. Ideally, the page live updates and and the user could scroll through the different visualizations and statistics. A challenge I see is that I want to have a slider scale for time inputs so users could instantly see the gradual changes in real time but all the instances I've seen slider used, it must be adjusted and submitted before the visualization is updated. This may be possible to overcome with a Javascript event listener.

### Exercise 3:
This CSV dataset is from the National Cancer Institute and its use is regulated by the Public Health Service Act. I removed all lines that did not include relevant data, removed the numbers after each state name, and removed the row representing Nevada because it had no useful data. My Flask wasn't working so I was not able to complete this exercise in time.

## Appendix
### Exercise 1
Failed/incomplete Code
```
import numpy as np
import matplotlib.pyplot as plt

def root(node_list):
  max_x = max(sublist[0] for sublist in node_list)
  max_y = max(sublist[1] for sublist in node_list)
  min_x = min(sublist[0] for sublist in node_list)
  min_y = min(sublist[1] for sublist in node_list)
  if max_x % 2 == 1: 
    max_x += 1
  if max_y % 2 == 1: 
    max_y += 1
  w = max_x - min_x
  h = max_y - min_y
  return min_x, min_y, max_x, max_y, w, h

class QuadTree():
    def __init__(self, min_x, min_y, max_x, max_y, w, h, k, node_list):
        x, y, knnclass = (sublist[0] for sublist in node_list), (sublist[1] for sublist in node_list), (sublist[2] for sublist in node_list) 
        self.x0 = min_x
        self.x1 = max_x
        self.y0 = min_y
        self.y1 = max_y
        self.width = w
        self.height = h
        k = len(node_list)
        self.children = []

    def divide(k, w, h, max_x, max_y, min_x, min_y):
        if k > 2:
          mid_x_ = (w / 2) + min_x
          mid_y_ = (h / 2) + min_y
          
          p = contains(min_x, min_y, mid_x_, mid_y_)
          #x1 = Node(min_x, min_y, mid_x_, mid_y_, p) #reset four corners of box
          divide(x1, k)

          p = contains(min_x, min_y+mid_y_, mid_x_, mid_y_)
          #x2 = Node(min_x, min_y+mid_y_, mid_x_, mid_y_, p)
          divide(x2, k)

          p = contains(min_x+mid_x_, min_y, mid_x_, mid_y_)
          #x3 = Node(min_x + mid_x_, min_y, mid_x_, mid_y_, p)
          divide(x3, k)

          p = contains(min_x+mid_x_, min_y+mid_y_, mid_x_, mid_y_)
          #x4 = Node(min_x+mid_x_, min_y+mid_y_, mid_x_, mid_y_, p)
          divide(x4, k)



def contains(x, y, w, h, node_list):
   pts = []
   for point in node_list:
       if point.x >= x and point.x <= x+w and point.y>=y and point.y<=y+h:
           pts.append(point)
   return pts
   
def euclidean_distance(self, x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

#(x, y; c)
```
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

