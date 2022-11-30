## BIS634 Assignment 4
### Goldie Zhu

### Exercise 1
Function to identify PubMed IDs
```
def get_pmids(condition, year):
    pmids = []
    r = requests.get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={condition}+AND+{year}[pdat]&retmode=xml&retmax=1000"
    )

```


### Exercise 2
### Exercise 3 - Fibonacci

#### recursive
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
### Appendix
