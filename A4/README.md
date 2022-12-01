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
    # have an extra row at top and extra col on left
    matrix = [ [] for i in range(length_m+1)] 
    compute_val = 0
    max_score = 0
    max_score_m = 0
    max_score_n = 0
    
    # init score in grid to zero
    matrix = np.zeros((length_m, length_n))
    
    ### First half of algo: Make the Matrix 
    for m in range(1, length_m):
        for n in range(1, length_n):
            # if match found
            if (seq1[m] == seq2[n]):
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
        if (matrix[tb_m-1][tb_n-1] == matrix[tb_m][tb_n] - match):
            match_seq1 = seq1[tb_m] + match_seq1
            match_seq2 = seq2[tb_n] + match_seq2
            # shift up and to the left
            tb_m -= 1 
            tb_n -= 1
        # If not a match (prioritize gaps, not mismatches)
        else:
            # current = l - gap
            if (matrix[tb_m][tb_n] == matrix[tb_m][tb_n-1] - gap_penalty):
                match_seq1 = '-' + match_seq1
                match_seq2 = seq2[tb_n] + match_seq2
                # shift left
                tb_n-= 1
            # current = up - gap
            elif (matrix[tb_m][tb_n] == matrix[tb_m-1][tb_n] - gap_penalty):
                match_seq1 = seq1[tb_m] + match_seq1
                match_seq2 = '-' + match_seq2
                # shift up
                tb_m -= 1
            else:
                tb_m -= 1 
                tb_n -= 1
    return match_seq1, match_seq2, max_score  
        
sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac')
print(sequence1)
print(sequence2)
print("score: ", score)

sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2)
print(sequence1)
print(sequence2)
print("score: ", score)

sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=8)
print(sequence1)
print(sequence2)
print("score: ", score)

sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', match = 2, gap_penalty=2)
print(sequence1)
print(sequence2)
print("score: ", score)


sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', match = 1, gap_penalty=2, mismatch_penalty = 2)
print(sequence1)
print(sequence2)
print("score: ", score)


sequence1, sequence2, score = smith_waterman('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', match = 5, gap_penalty=2, mismatch_penalty = 2)
print(sequence1)
print(sequence2)
print("score: ", score)
```
Solutions:
```
accct-ac-tgac
gacctagctcgac
score:  8.0
gcatcga
gcatcga
score:  7.0
gcatcga
gcatcga
score:  7.0
cagaccct-ac-tgac
ca-gacctagctcgac
score:  18.0
gcatcga
gcatcga
score:  7.0
catcgagaccct-ac-tgac
c--t--agacctagctcgac
score:  56.0
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
```

