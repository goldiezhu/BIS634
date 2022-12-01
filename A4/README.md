## BIS634 Assignment 4
### Goldie Zhu

### Exercise 1
2D Gradient Descent
```
import requests

def error(a = 0.1, b = 0.1):
    return float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}", headers={"User-Agent": "MyScript"}).text)

def optimize(a, b, h):
    deriv_a = a - 0.1 * ((error(a + h,b) - error(a,b))/h)
    deriv_b = b - 0.1 * ((error(a,b + h) - error(a,b))/h)
    while abs(error(deriv_a, deriv_b) - error(a,b)) > 0.0001:
        a, b = deriv_a, deriv_b
        deriv_a = deriv_a - 0.1 * ((error(a + h,b) - error(a,b))/h)
        deriv_b = deriv_b - 0.1 * ((error(a,b + h) - error(a,b))/h)
    return deriv_a, deriv_b

optimize(0.2, 0.3, 0.00001)
```
I estimate the gradient using the following two equations, which combined makes the third equation. The equation uses small values of h to get closer to the desired approximation of the derivative.

<img width="242" alt="Screen Shot 2022-11-30 at 6 39 51 PM" src="https://user-images.githubusercontent.com/37753494/204931469-362d58dd-0e52-4216-ae72-71320ba2f774.png">
<img width="242" alt="Screen Shot 2022-11-30 at 6 39 56 PM" src="https://user-images.githubusercontent.com/37753494/204931470-b44fc796-ea33-40a4-8da8-17211de07bfb.png">

<img width="514" alt="Screen Shot 2022-11-30 at 6 47 14 PM" src="https://user-images.githubusercontent.com/37753494/204932281-9f4865a3-1ab6-47e2-ad86-464b1682e747.png">

I chose 0.0001 as the stopping criteria because I only needed a sufficiently small number. The steps are set to 0.1 because that is a standard step size and it doesn't have to be too small or else the calculation will take too long. I tested h for different values, with the closest being 1e-10. 

<img width="270" alt="Screen Shot 2022-11-30 at 6 50 02 PM" src="https://user-images.githubusercontent.com/37753494/204932614-fd15d28a-a7ca-49fd-aee4-4d1a1ef8a141.png">


To find the local and global minimums, I find the optimized points and then find the error for each of those points. I plotted the points, which resulted in a global minimum of less than 1.01 and local minimums of around 1.11. If I didn't know how many minimas there are, I could run a large n number of tests for clusters, such as the one around 1.02.
```
import numpy as np

minimums = []
for i in np.linspace(0.1,1,5, endpoint = False):
    for j in np.linspace(0.1,1,5, endpoint = False):
        min = optimize(i,j, 0.00000000001)
        minimums.append(min)
for k in minimums:
    print(error(k[0], k[1]))

```
<img width="425" alt="Screen Shot 2022-11-30 at 7 43 56 PM" src="https://user-images.githubusercontent.com/37753494/204938602-7551e18f-fb68-4a59-9b3c-97ff96a60249.png">


### Exercise 2
Haversine
```
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
```
Plot
```
import pandas as pd
import plotnine as p9
import random
import numpy as np
k=3
df = pd.read_csv('worldcities.csv')
def normalize(series):
    return (series - series.mean()) / series.std()

pts = [np.array(pt) for pt in zip(df['lat'], df['lng'])]
centers = random.sample(pts, k)
old_cluster_ids, cluster_ids = None, [] # arbitrary but different
while cluster_ids != old_cluster_ids:
    old_cluster_ids = list(cluster_ids)
    cluster_ids = []
    for pt in pts:
        min_cluster = -1
        min_dist = float('inf')
        for i, center in enumerate(centers):
            dist = np.linalg.norm(pt - center)
            if dist < min_dist:
                min_cluster = i
                min_dist = dist
        cluster_ids.append(min_cluster)
    df['cluster'] = cluster_ids
    cluster_pts = [[pt for pt, cluster in zip(pts, cluster_ids) if cluster == match]
        for match in range(k)]
    centers = [sum(pts)/len(pts) for pts in cluster_pts]
(p9.ggplot(df, p9.aes(x="lat", y="lng", color="cluster")) 
    + p9.geom_point()).draw()
```
<img width="573" alt="Screen Shot 2022-11-30 at 9 19 36 PM" src="https://user-images.githubusercontent.com/37753494/204950275-ce69a67d-4595-4004-a8f2-8325be3d7e30.png">

Coordinate data
```
coordinate_pairs = []
for cluster in range(k):
  clusters = [np.array(pt) for pt in zip(df[df['cluster']==cluster]['lat'], df[df['cluster']== cluster]['lng'])]
  coordinate_pairs.append(clusters)
```

Cartopy plot
```
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import random 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = [pt[0] for pt in pts]
lngs = [pt[1] for pt in pts]
ax.coastlines()
for j in range(k):
  lats = [pt[0] for pt in coordinate_pairs[j]]
  lngs = [pt[1] for pt in coordinate_pairs[j]]
  ax.plot(lngs, lats, "o", transform=ccrs.PlateCarree())
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
plt.show()
```

k = 3

<img width="298" alt="Screen Shot 2022-11-30 at 9 20 49 PM" src="https://user-images.githubusercontent.com/37753494/204950405-69b226d0-78a7-45ec-b317-cd0c754fa054.png">

<img width="297" alt="Screen Shot 2022-11-30 at 9 22 43 PM" src="https://user-images.githubusercontent.com/37753494/204950656-1038703e-b268-46da-991b-92157f6f78f5.png">

<img width="295" alt="Screen Shot 2022-11-30 at 9 23 44 PM" src="https://user-images.githubusercontent.com/37753494/204950747-ec9863be-9075-4c7b-b58f-4c57bfd840e5.png">

k = 5

<img width="503" alt="Screen Shot 2022-11-30 at 9 27 16 PM" src="https://user-images.githubusercontent.com/37753494/204951194-b5bb7865-3fc3-4b6e-abdc-412ea6d23c5a.png">
<img width="261" alt="Screen Shot 2022-11-30 at 9 27 34 PM" src="https://user-images.githubusercontent.com/37753494/204951224-2599780e-41f4-4297-8dea-3f38fba4db6f.png">

<img width="258" alt="Screen Shot 2022-11-30 at 9 28 56 PM" src="https://user-images.githubusercontent.com/37753494/204951406-2eec5586-8996-4be9-abeb-0e27412c5246.png">
<img width="260" alt="Screen Shot 2022-11-30 at 9 30 29 PM" src="https://user-images.githubusercontent.com/37753494/204951633-aaf7b2ee-31df-4979-a1e2-954ed172e16c.png">

k=7

<img width="501" alt="Screen Shot 2022-11-30 at 9 30 46 PM" src="https://user-images.githubusercontent.com/37753494/204951707-0d37f69b-9b80-4b70-8fcb-d59ec3289aae.png">
<img width="261" alt="Screen Shot 2022-11-30 at 9 30 58 PM" src="https://user-images.githubusercontent.com/37753494/204951708-b94f097f-6b2c-4144-a68b-b9302a3c8bea.png">

<img width="259" alt="Screen Shot 2022-11-30 at 9 32 23 PM" src="https://user-images.githubusercontent.com/37753494/204951895-ec8ef962-2512-4198-84cd-b0f2d7000f78.png">

<img width="258" alt="Screen Shot 2022-11-30 at 9 33 29 PM" src="https://user-images.githubusercontent.com/37753494/204952053-0f7663fb-4755-4808-a3c4-eddcdb1522b4.png">

k=15

<img width="495" alt="Screen Shot 2022-11-30 at 9 33 57 PM" src="https://user-images.githubusercontent.com/37753494/204952100-cda9ef9f-bc49-4da8-ad46-9cfee2a1b865.png">

<img width="257" alt="Screen Shot 2022-11-30 at 9 34 22 PM" src="https://user-images.githubusercontent.com/37753494/204952165-7604e741-dd00-4bc5-b32c-a2ad120889f5.png">
<img width="257" alt="Screen Shot 2022-11-30 at 9 34 30 PM" src="https://user-images.githubusercontent.com/37753494/204952166-3740b30f-45eb-495f-8e18-458d2e16854e.png">
<img width="257" alt="Screen Shot 2022-11-30 at 9 37 09 PM" src="https://user-images.githubusercontent.com/37753494/204952559-1217e6f2-65fe-4bfa-9f7f-12ac991bc1e7.png">
<img width="258" alt="Screen Shot 2022-11-30 at 9 41 52 PM" src="https://user-images.githubusercontent.com/37753494/204953271-e83f15fc-e67e-46a9-901b-bf625ee17a91.png">

There seems to be more diversity and more complexly shaped clusters as k gets larger. When k was little, the cluster shapes remained basicallythe same. However, my k=15 did not show the full diversity and extend of the clusters so it is not representative of how k=15 should've looked..


### Exercise 3 - Fibonacci

```
# The n-th Fibonacci number is the sum of the (n-1)th and (n-2)th
from functools import lru_cache
import time
import pandas as pd
import plotnine as p9
from tqdm import tqdm

### Recursive Strategy ###
def fib_rec(n):
    if (n <= 1):
        return n
    else:
        return(fib_rec(n-1) + fib_rec(n-2))

def timeit(function, *args, n=3):
    times = []
    for i in range(n):
        start = time.time()
        function(*args)
        times.append(time.time() - start)
    return min(times)

# The n-th Fibonacci number is the sum of the (n-1)th and (n-2)th

### LRU Cache ###
@lru_cache()
def fib_cache(n):
    if n in {1,2}:
        return n
    return(fib_cache(n-1) + fib_cache(n-2))

def timeit(function, *args, n=3):
    times = []
    for i in range(n):
        start = time.time()
        function(*args)
        times.append(time.time() - start)
    return min(times)

ns = range(1,40)
times = [timeit(fib_cache,n) for n in tqdm(ns)]
times1 = [timeit(fib_rec,n) for n in tqdm(ns)]


df = pd.DataFrame({'n':ns, 'time (s)': times, 'time1 (s)': times1})
df['x'] = df.index
p9.ggplot(p9.aes(x='n'), data=df) +\
    p9.geom_line(p9.aes(y='time (s)'), color='blue') +\
    p9.geom_line(p9.aes(y='time1 (s)'), color='red') 
```
<img width="472" alt="Screen Shot 2022-11-30 at 5 27 25 PM" src="https://user-images.githubusercontent.com/37753494/204921848-1b2a8bce-eeb8-4221-82ec-7bad015aae74.png">

I chose n = 40 because I needed a large number that will show the difference in time between normal recursion and cache recursion. However, the time for cache is relatively consistent so I could have used a smaller n and the recursive time would still should an exponential slope. The graph is time vs n rather than using the fibonacci number or any other number because the fibonacci number itself does not matter. The results indicate that up until around n = 30, the time for recursion and cache fibonacci are around the same. After a bit before n=30, the recursive fibonacci starts getting exponentially slower.

### Exercise 4
Smith-Waterman function
```

# Smith - Watemran Algorithm
import numpy as np

### PART 1 ###
# implement function that takes two strings and uses the Smith-Waterman Algorithm
# to return an optimal local alignment and score
# insert '-' to indicate gap 
# take three keyword arguments with default 1 
# (penalty of one applied to match scores for each missing or changed letter)

    
def smith_waterman(seq1, seq2, match = 1, gap_penalty = 1, mismatch_penalty = 1):
    length_m = len(seq1)
    length_n = len(seq2)
    compute_val = 0
    max_score = 0
    max_score_m = 0
    max_score_n = 0
    
   # init matrix to zero, have an extra row at top and extra col on left
    matrix = np.zeros((length_m+1, length_n+1), np.int)
    ### First half of algo: Make the Matrix 
    for m in range(1, length_m+1):
        for n in range(1, length_n+1):
            # if match found
            if (seq1[m-1] == seq2[n-1]):
                # upper left + match
                compute_val = matrix[m-1][n-1] + match
            # if match not found
            else:
                # upper left - mismatch penalty
                compute_val = matrix[m-1][n-1] - mismatch_penalty

            # find actual value to put into matrix
            matrix[m][n] = max(compute_val, matrix[m][n-1] - gap_penalty, matrix[m-1][n] - gap_penalty, 0)
            
            # check max score
            if (matrix[m][n] > max_score):
                max_score = matrix[m][n]
                # add 1 to max scores to account for the 0 index
                max_score_m = m
                max_score_n = n
   
    ### Second half of algo: Backtracking from max value 
    # prioritizing gap (insertion and deletions), and not mismatch
    
    # corresponding seq element from max value
    tb_m = max_score_m
    tb_n = max_score_n
    match_seq1 = ""
    match_seq2 = ""
    while (matrix[tb_m][tb_n] > 0):
        if ((seq1[tb_m-1] == seq2[tb_n-1]) and matrix[tb_m-1][tb_n-1] == matrix[tb_m][tb_n] - match):
            match_seq1 = seq1[tb_m-1] + match_seq1
            match_seq2 = seq2[tb_n-1] + match_seq2
            # shift up and to the left
            tb_m -= 1 
            tb_n -= 1
        # If not a match (prioritize gaps, not mismatches)
        else:
            # current = l - gap
            if (matrix[tb_m][tb_n] == matrix[tb_m][tb_n-1] - gap_penalty):
                match_seq1 = '-' + match_seq1
                match_seq2 = seq2[tb_n-1] + match_seq2
                # shift left
                tb_n-= 1
            # current = up - gap
            elif (matrix[tb_m][tb_n] == matrix[tb_m-1][tb_n] - gap_penalty):
                match_seq1 = seq1[tb_m-1] + match_seq1
                match_seq2 = '-' + match_seq2
                # shift up
                tb_m -= 1
            else:
                tb_m -= 1 
                tb_n -= 1
    return match_seq1, match_seq2, max_score  


### PART 2 ###
# Test it, and explain how tests show the function works. Test other values.

# Examples from the problem statement:
sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac')
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))

# Example from the cheatsheet
sequence1, sequence2, score = smith_waterman('gttacc', 'gttgac')
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))

# To test whether or not the above smith-waterman function is correct, I will manipulate the parameters.

# Here is the control:
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 5, gap_penalty = 4, mismatch_penalty = 3)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))

# Now I wil increase the match value. If I increase the match value, I expect score to increase because I'm rewarding more for matching nucleotides.
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 6, gap_penalty = 4, mismatch_penalty = 3)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
# The score increased from 18 to 23.

# Starting from the control again, I will now increase only the gap_penalty value. This should decrease the score.
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 5, gap_penalty = 5, mismatch_penalty = 3)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
# The score decreased from 18 to 17.

# Starting from the control again, I will now increase only the mismatch_penalty value. This should also decrease the score.
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 5, gap_penalty = 4, mismatch_penalty = 4)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
# The score decreased from 18 to 17.

```
Solutions:
```
Sequence 1: agacccta-ct-gac, Sequence 2: aga-cctagctcgac, Score: 8
Sequence 1: gcatcga, Sequence 2: gcatcga, Score: 7
Sequence 1: gtt-ac, Sequence 2: gttgac, Score: 4
Sequence 1: gatt-a, Sequence 2: gattca, Score: 18
Sequence 1: gatt-a, Sequence 2: gattca, Score: 23
Sequence 1: gatt, Sequence 2: gatt, Score: 17
Sequence 1: gatt-a, Sequence 2: gattca, Score: 17
```

### Appendix

Exercise 1
```
import requests

def error(a = 0.1, b = 0.1):
    return float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}", headers={"User-Agent": "MyScript"}).text)

def optimize(a, b, h):
    deriv_a = a - 0.1 * ((error(a + h,b) - error(a,b))/h)
    deriv_b = b - 0.1 * ((error(a,b + h) - error(a,b))/h)
    while abs(error(deriv_a, deriv_b) - error(a,b)) > 0.0001:
        a, b = deriv_a, deriv_b
        deriv_a = deriv_a - 0.1 * ((error(a + h,b) - error(a,b))/h)
        deriv_b = deriv_b - 0.1 * ((error(a,b + h) - error(a,b))/h)
    return deriv_a, deriv_b

optimize(0.2, 0.3, 0.00001)
optimize(0.2, 0.3, 0.00000000001)

import numpy as np

minimums = []
for i in np.linspace(0.1,1,5, endpoint = False):
    for j in np.linspace(0.1,1,5, endpoint = False):
        min = optimize(i,j, 0.00000000001)
        minimums.append(min)
for k in minimums:
    print(error(k[0], k[1]))

```

Exercise 2
```
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
    
import pandas as pd
import plotnine as p9
import random
import numpy as np
k=3
df = pd.read_csv('worldcities.csv')
def normalize(series):
    return (series - series.mean()) / series.std()

pts = [np.array(pt) for pt in zip(df['lat'], df['lng'])]
centers = random.sample(pts, k)
old_cluster_ids, cluster_ids = None, [] # arbitrary but different
while cluster_ids != old_cluster_ids:
    old_cluster_ids = list(cluster_ids)
    cluster_ids = []
    for pt in pts:
        min_cluster = -1
        min_dist = float('inf')
        for i, center in enumerate(centers):
            dist = np.linalg.norm(pt - center)
            if dist < min_dist:
                min_cluster = i
                min_dist = dist
        cluster_ids.append(min_cluster)
    df['cluster'] = cluster_ids
    cluster_pts = [[pt for pt, cluster in zip(pts, cluster_ids) if cluster == match]
        for match in range(k)]
    centers = [sum(pts)/len(pts) for pts in cluster_pts]
(p9.ggplot(df, p9.aes(x="lat", y="lng", color="cluster")) 
    + p9.geom_point()).draw()
    
coordinate_pairs = []
for cluster in range(k):
  clusters = [np.array(pt) for pt in zip(df[df['cluster']==cluster]['lat'], df[df['cluster']== cluster]['lng'])]
  coordinate_pairs.append(clusters)
  
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import random 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
lats = [pt[0] for pt in pts]
lngs = [pt[1] for pt in pts]
ax.coastlines()
for j in range(k):
  lats = [pt[0] for pt in coordinate_pairs[j]]
  lngs = [pt[1] for pt in coordinate_pairs[j]]
  ax.plot(lngs, lats, "o", transform=ccrs.PlateCarree())
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
plt.show()

#change k and rerun code for different ks
```

Exercise 3
```

# The n-th Fibonacci number is the sum of the (n-1)th and (n-2)th
from functools import lru_cache
import time
import pandas as pd
import plotnine as p9
from tqdm import tqdm

### Recursive Strategy ###
def fib_rec(n):
    if (n <= 1):
        return n
    else:
        return(fib_rec(n-1) + fib_rec(n-2))

def timeit(function, *args, n=3):
    times = []
    for i in range(n):
        start = time.time()
        function(*args)
        times.append(time.time() - start)
    return min(times)

# The n-th Fibonacci number is the sum of the (n-1)th and (n-2)th

### LRU Cache ###
@lru_cache()
def fib_cache(n):
    if n in {1,2}:
        return n
    return(fib_cache(n-1) + fib_cache(n-2))

def timeit(function, *args, n=3):
    times = []
    for i in range(n):
        start = time.time()
        function(*args)
        times.append(time.time() - start)
    return min(times)

ns = range(1,40)
times = [timeit(fib_cache,n) for n in tqdm(ns)]
times1 = [timeit(fib_rec,n) for n in tqdm(ns)]


df = pd.DataFrame({'n':ns, 'time (s)': times, 'time1 (s)': times1})
df['x'] = df.index
p9.ggplot(p9.aes(x='n'), data=df) +\
    p9.geom_line(p9.aes(y='time (s)'), color='blue') +\
    p9.geom_line(p9.aes(y='time1 (s)'), color='red') 
```
Exercise 4
```

# Smith - Watemran Algorithm
import numpy as np

### PART 1 ###
# implement function that takes two strings and uses the Smith-Waterman Algorithm
# to return an optimal local alignment and score
# insert '-' to indicate gap 
# take three keyword arguments with default 1 
# (penalty of one applied to match scores for each missing or changed letter)

    
def smith_waterman(seq1, seq2, match = 1, gap_penalty = 1, mismatch_penalty = 1):
    length_m = len(seq1)
    length_n = len(seq2)
    compute_val = 0
    max_score = 0
    max_score_m = 0
    max_score_n = 0
    
   # init matrix to zero, have an extra row at top and extra col on left
    matrix = np.zeros((length_m+1, length_n+1), np.int)
    ### First half of algo: Make the Matrix 
    for m in range(1, length_m+1):
        for n in range(1, length_n+1):
            # if match found
            if (seq1[m-1] == seq2[n-1]):
                # upper left + match
                compute_val = matrix[m-1][n-1] + match
            # if match not found
            else:
                # upper left - mismatch penalty
                compute_val = matrix[m-1][n-1] - mismatch_penalty

            # find actual value to put into matrix
            matrix[m][n] = max(compute_val, matrix[m][n-1] - gap_penalty, matrix[m-1][n] - gap_penalty, 0)
            
            # check max score
            if (matrix[m][n] > max_score):
                max_score = matrix[m][n]
                # add 1 to max scores to account for the 0 index
                max_score_m = m
                max_score_n = n
   
    ### Second half of algo: Backtracking from max value 
    # prioritizing gap (insertion and deletions), and not mismatch
    
    # corresponding seq element from max value
    tb_m = max_score_m
    tb_n = max_score_n
    match_seq1 = ""
    match_seq2 = ""
    while (matrix[tb_m][tb_n] > 0):
        if ((seq1[tb_m-1] == seq2[tb_n-1]) and matrix[tb_m-1][tb_n-1] == matrix[tb_m][tb_n] - match):
            match_seq1 = seq1[tb_m-1] + match_seq1
            match_seq2 = seq2[tb_n-1] + match_seq2
            # shift up and to the left
            tb_m -= 1 
            tb_n -= 1
        # If not a match (prioritize gaps, not mismatches)
        else:
            # current = l - gap
            if (matrix[tb_m][tb_n] == matrix[tb_m][tb_n-1] - gap_penalty):
                match_seq1 = '-' + match_seq1
                match_seq2 = seq2[tb_n-1] + match_seq2
                # shift left
                tb_n-= 1
            # current = up - gap
            elif (matrix[tb_m][tb_n] == matrix[tb_m-1][tb_n] - gap_penalty):
                match_seq1 = seq1[tb_m-1] + match_seq1
                match_seq2 = '-' + match_seq2
                # shift up
                tb_m -= 1
            else:
                tb_m -= 1 
                tb_n -= 1
    return match_seq1, match_seq2, max_score  


### PART 2 ###
# Test it, and explain how tests show the function works. Test other values.

# Examples from the problem statement:
sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac')
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))

# Example from the cheatsheet
sequence1, sequence2, score = smith_waterman('gttacc', 'gttgac')
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))

# To test whether or not the above smith-waterman function is correct, I will manipulate the parameters.

# Here is the control:
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 5, gap_penalty = 4, mismatch_penalty = 3)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))

# Now I wil increase the match value. If I increase the match value, I expect score to increase because I'm rewarding more for matching nucleotides.
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 6, gap_penalty = 4, mismatch_penalty = 3)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
# The score increased from 18 to 23.

# Starting from the control again, I will now increase only the gap_penalty value. This should decrease the score.
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 5, gap_penalty = 5, mismatch_penalty = 3)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
# The score decreased from 18 to 17.

# Starting from the control again, I will now increase only the mismatch_penalty value. This should also decrease the score.
sequence1, sequence2, score = smith_waterman('gacttac', 'cgtgaattcat', match = 5, gap_penalty = 4, mismatch_penalty = 4)
print('Sequence 1: {}, Sequence 2: {}, Score: {}'.format(sequence1, sequence2, score))
# The score decreased from 18 to 17.

```

