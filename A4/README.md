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

