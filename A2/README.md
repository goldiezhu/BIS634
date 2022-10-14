## BIS634 Assignment 2
### Goldie Zhu

### Exercise 1
1. THE friend in exercise 1 is trying to load too much data with their code, which is causing the MemoryError. The program needs more than 4 GB and even 8 GB to store all the data in the list. 
2. Instead of using a list, the friend can use an array, which uses significantly less bytes. This is because an array does not need to store metadata, such as value, data type, etc. 
3. A strategy to calculate the average without storing all the data is to use one value to store the sum of all the values instead of storing all the values. Then, you can divide by the total number of people to get the average.

### Exercise 2
1. Implementing a Bloom Filter from scratch
I have a list of bloom filters for different purposes. 'data' is for the first part where I implement a bloom filter from scratch but it does the same job as data 3, which implements all 3 hashes. data 1 and data 2 respectively implement 1 and 2 hashes, which are used in the later parts of the problem. The entire bitarray is first set to False, which means there is no data in the bitarray.
```
data = bitarray.bitarray(size)
data[:] = False
data1 = bitarray.bitarray(size)
data1[:] = False
data2 = bitarray.bitarray(size)
data2[:] = False
data3 = bitarray.bitarray(size)
data3[:] = False
```
Each word is put through the 3 hashes and the location in the bitarray is changed from False to True. 
```
hash1 = my_hash(word)
hash2 = my_hash2(word)
hash3 = my_hash3(word)
data[hash1] = True
data[hash2] = True
data[hash3] = True
```
2. Suggest Spelling Corrections
First, input a word to suggest spelling suggestions for. This inputted word is stored as a list in 'check_word.'
```
check_word = list(input("Input a word for spelling suggestions: "))
```
This list is compared with another list 'alphabet_list', which contains every character of the alphabet. With two 'for' loops, each character in the inputted word is switched with every letter in the alphabet. After each change, the word 'temp' is put through the hashes and the word is checked to see if it is in the bloom filter, which already had words inputted in the implementation.
```
for i in range(len(check_word)):
  for j in range(alphabet_count):
    temp = check_word.copy()
    temp[i] = alphabet_list[j]
    altered_word = "".join(temp)
    check2(altered_word)
```
If the word is in the bloom filter, all three of the locations indicated by the hashes should be True. If the word is in the bloom filter, it is added to a list.
```
if data3[check_hash1] == True and data3[check_hash2] == True and data3[check_hash3] == True:
    possible_words3.append(check_existence)
```
I have three lists that correlate with using one hash, two hashes, and three hashes. The output will show all spelling suggestions for the hash groups.

3. typos.json comparison
First, I made a list of lists and added all the terms in typos.json while filtering out all the correct terms in the typos.json list.
```
for i in range(len(orig_typos_list)):
  if orig_typos_list[i][0] != orig_typos_list[i][1]:
    typos_list.append(orig_typos_list[i])
```
Then, I go through the list of typos and check if the first (incorrect) word is in my bloom filter while replacing every letter in the word with every character in the alphabet. If the second (correct) word appears in the suggestions and there are less than three suggestions, I add to a hashcount, which keeps track of how many words meet the previously mentioned criteria for one hash, two hashes, and three hashes respectively. This hash count is divided by the number of incorrect words in the typo_list to calculate how many bits are necessary to give good suggestions 90% of the time. It takes approximately 1.7e8 bytes for one hash, 1.7e7 bytes for two hashes, and 9.3e6 bytes for three hashes.

### Exercise 3
Below is the **add** method where it checks if the entered value should go to the left or right of the current node:
```
def add(self, value):
    if self._value is None:
      self._value = value
      return
    if value == self._value:
      return
    if value < self._value:
      if self.left:
        self.left.add(value)
      else:
        self.left = Tree(value)
    else:
      if self.right:
        self.right.add(value)
      else:
        self.right = Tree(value)
```
Below is loglog plot that demonstrates in is executing in O(log n) times: 

<img width="388" alt="Screen Shot 2022-10-14 at 5 40 44 PM" src="https://user-images.githubusercontent.com/37753494/195948238-d9a8dbe0-2e3c-459b-a7a7-9c8e47378f4e.png">

Below is loglog plot that shows the tree is O(n log n) and the runtime lies between a curve that is O(n) and one that is O(n**2):
<img width="390" alt="Screen Shot 2022-10-14 at 5 42 01 PM" src="https://user-images.githubusercontent.com/37753494/195948352-cb29fb79-3bdd-4ba6-825c-75a97d6bf22b.png">

### Exercise 4

<img width="434" alt="Screen Shot 2022-10-14 at 5 51 44 PM" src="https://user-images.githubusercontent.com/37753494/195949390-4af63416-757d-4914-864b-a08276ab8c82.png">

Hypothesis: The two functions order the list from least to greatest.

alg1 iterates thru a list of numbers and arranges them from least to greatest. After making sure that the input is a list, the boolean value "changes" makes sure that the "while" function hasn't reached the end of the list yet. The for loop in the while loop switches the places of two numbers if the second number is smaller than the first one. If "changes" is False, then the while loop ends and it returns the ordered list.

alg2 first checks if the inputted list is longer than one number. If it is, then the inputted list is split into two halves. 'left' and 'right' are iterators that iterate thru their respective halves of the split. While the iterators haven't reached the end of the list (False), the values of the left and right halves are compared and the smaller one is appended to the results list. If the two halves are not the same size, the loop will reach 'StopIteration,' which will exit the loop and return the ordered list.

** alg2 is better.**

Below is the log-log graph timing the performance of alg1 and alg2:
This plot has all algs and data sets.

<img width="416" alt="Screen Shot 2022-10-14 at 5 45 48 PM" src="https://user-images.githubusercontent.com/37753494/195948808-e04553fb-6b56-4d6c-9a32-8059e547a212.png">


The following three plots compare the performace of alg1 and alg2 across the three datasets.

<img width="406" alt="Screen Shot 2022-10-14 at 5 45 52 PM" src="https://user-images.githubusercontent.com/37753494/195948836-e08cdea8-697d-4d7f-8382-3b6ba6feee0f.png">

The big-O scaling indicates that alg1 is much slower than alg2. 

<img width="422" alt="Screen Shot 2022-10-14 at 5 45 57 PM" src="https://user-images.githubusercontent.com/37753494/195948852-f48e5b46-350b-44d6-ab94-d15e56213b6f.png">
alg1 performs better than alg2 with data2.
<img width="447" alt="Screen Shot 2022-10-14 at 5 46 02 PM" src="https://user-images.githubusercontent.com/37753494/195948857-6bbfc423-2ef4-47b1-8b4a-9fc6d85c0740.png">
alg2 performs better than alg1 with data3.

This plot compares the performace of alg1 for both datasets.

<img width="388" alt="Screen Shot 2022-10-14 at 5 46 06 PM" src="https://user-images.githubusercontent.com/37753494/195948904-71a27163-302e-4f5c-bcb6-ff07cca3003d.png">

Conclusion: alg1 seems to be O(n**2) and alg2 seems to be O(n*logn). alg2 would perform much better for arbitrary data because it is faster at sorting and shows more consistent good performance.

Parallelize Explanation: To parallelize alg2, you must find tasks that can be done at the same time. In this case, there aren't independent tasks/functions so it is better to breakdown the dataset and run the ordering simultaneously. The ordering of the left and the right half of a dataset can be independent and after the ordering is finished, you can merge the two ordered halves.

### Appendix
Exercise 2
```
import bitarray
from hashlib import sha3_256, sha256, blake2b

size = int(1e7)

def my_hash(s):
  return int(sha256(s.lower().encode()).hexdigest(), 16) % size

def my_hash2(s):
  return int(blake2b(s.lower().encode()).hexdigest(), 16) % size

def my_hash3(s):
  return int(sha3_256(s.lower().encode()).hexdigest(), 16) % size

def check():
  check_existence = input("Input a word to see if it is in the Bloom Filter: ")
  check_hash1 = my_hash(check_existence)
  check_hash2 = my_hash2(check_existence)
  check_hash3 = my_hash3(check_existence)
  if data[check_hash1] == False or data[check_hash2] == False or data[check_hash3] == False:
    print(check_existence, "is not in the filter.")
  else:    
    print(check_existence, "is in the filter.")
    

data = bitarray.bitarray(size)
data[:] = False
data1 = bitarray.bitarray(size)
data1[:] = False
data2 = bitarray.bitarray(size)
data2[:] = False
data3 = bitarray.bitarray(size)
data3[:] = False

with open('words.txt') as f:
  for line in f:
    word = line.strip()
    hash1 = my_hash(word)
    hash2 = my_hash2(word)
    hash3 = my_hash3(word)
    data[hash1] = True
    data[hash2] = True
    data[hash3] = True

    data1[hash1] = True

    data2[hash1] = True
    data2[hash2] = True

    data3[hash1] = True
    data3[hash2] = True
    data3[hash3] = True

check()
```
```
import string
!pip install ascii
import ascii
import copy

check_word = list(input("Input a word for spelling suggestions: "))
alphabet = string.ascii_lowercase
alphabet_list = [i for i in alphabet]
alphabet_count = 26
possible_words1 = []
possible_words2 = []
possible_words3 = []

def check2(check_existence):
  check_hash1 = my_hash(check_existence)
  check_hash2 = my_hash2(check_existence)
  check_hash3 = my_hash3(check_existence)

  if data1[check_hash1] == True:
    possible_words1.append(check_existence)
  
  if data2[check_hash1] == True and data2[check_hash2] == True:
    possible_words2.append(check_existence)

  if data3[check_hash1] == True and data3[check_hash2] == True and data3[check_hash3] == True:
    possible_words3.append(check_existence)

def print_suggestions():
  print("One Hash")
  print(possible_words1)
  print("Two Hash")
  print(possible_words2)
  print("Three Hash")
  print(possible_words3)

for i in range(len(check_word)):
  for j in range(alphabet_count):
    temp = check_word.copy()
    temp[i] = alphabet_list[j]
    altered_word = "".join(temp)
    check2(altered_word)

print_suggestions()
```
```
import json
with open("typos.json") as f:
  orig_typos_list = json.load(f)

typos_list = []
hash1_suggestions = []
hash2_suggestions = []
hash3_suggestions = []

hash1_count = 0
hash2_count = 0
hash3_count = 0


def check3(check_existence):
  check_hash1 = my_hash(check_existence)
  check_hash2 = my_hash2(check_existence)
  check_hash3 = my_hash3(check_existence)

  if data1[check_hash1] == True:
    hash1_suggestions.append(check_existence)
  
  if data2[check_hash1] == True and data2[check_hash2] == True:
    hash2_suggestions.append(check_existence)

  if data3[check_hash1] == True and data3[check_hash2] == True and data3[check_hash3] == True:
    hash3_suggestions.append(check_existence)

def hash_counts(correct_word, hash1_count, hash2_count, hash3_count):

  if len(hash1_suggestions) <= 3:
    if correct_word in hash1_suggestions:
      hash1_count += 1

  if len(hash2_suggestions) <= 3:
    if correct_word in hash2_suggestions:
      hash2_count += 1

  if len(hash3_suggestions) <= 3:
    if correct_word in hash3_suggestions:
      hash3_count += 1

  hash1_suggestions.clear()
  hash2_suggestions.clear()
  hash3_suggestions.clear()

  return(hash1_count, hash2_count, hash3_count)  

#filter out correct terms
for i in range(len(orig_typos_list)):
  if orig_typos_list[i][0] != orig_typos_list[i][1]:
    typos_list.append(orig_typos_list[i])

#spelling suggestions
for i in range(len(typos_list)):
  specific_typos_string = typos_list[i][0]
  typos_string_list = [i for i in specific_typos_string]
  for j in range(len(typos_string_list)):
    for k in range(alphabet_count):
      temp = typos_string_list.copy()
      temp[j] = alphabet_list[k]
      altered_word = "".join(temp)
      check3(altered_word)
  hash1_count, hash2_count, hash3_count = hash_counts(typos_list[i][1], hash1_count, hash2_count, hash3_count)

print(hash1_count/len(typos_list))
print(hash2_count/len(typos_list))
print(hash3_count/len(typos_list))
```
```
import matplotlib.pyplot as plt
import numpy as np
size = ["10", "100", "1e3", "1e4", "1e5", "1e6", "1e7", "1e8", "1e9", "1e10"]
h1 = [0, 0, 0, 0, 0, 0, 0.00456, 0.81052, 0.9442, 0.94756]
h2 = [0, 0, 0, 0, 0, 0, 0.62092, 0.9478, 0.94808, 0.94808]
h3 = [0, 0, 0, 0, 0, 0, 0.916, 0.94808, 0.94808, 0.94808]
plt.plot(size, h1)
plt.plot(size, h2)
plt.plot(size, h3)
```
Exercise 3
```
class Tree:
  def __init__(self, value = None):
    self._value = value
    self.left = None
    self.right = None

  def __contains__(self, item):
    if self._value == item:
      return True
    elif self.left and item < self._value:
      return item in self.left
    elif self.right and item > self._value:
      return item in self.right
    else:
      return False

  def add(self, value):
    if self._value is None:
      self._value = value
      return
    if value == self._value:
      return
    if value < self._value:
      if self.left:
        self.left.add(value)
      else:
        self.left = Tree(value)
    else:
      if self.right:
        self.right.add(value)
      else:
        self.right = Tree(value)

my_tree = Tree()
for item in [55, 62, 37, 49, 71, 14, 17]:
  my_tree.add(item)
print(55 in my_tree)
print(42 in my_tree)
```
```
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import math
import random

ns = [2**i for i in range(24)]
find_time = []

for item in ns: #number of trees
  my_tree = Tree()
  for i in range(item): #length of trees
    my_tree.add(random.randint(0,10000))

  start = timer()
  for i in range(1000):
    i in my_tree
  end = timer()
  find_time.append((end - start)/1000)

plt.loglog(ns, find_time)
```
```
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import math
import random

ns = list(np.logspace(0, 5, dtype = int))
find_time = []
O_n = []
O_nsq = []

for item in ns: #number of trees
  my_tree = Tree()
  start = timer()
  for i in range(item): #length of trees
    my_tree.add(random.randint(0,10000))
  end = timer()
  find_time.append((end - start)*1000000)
    
for i in range(1, 100000):
    O_n.append(i*2)
    O_nsq.append(i**2)

plt.loglog(ns, find_time, label = 'O(nLogn)')
plt.plot(O_nsq, label = 'O(N^2)')
plt.plot(O_n, label = 'O(N)')

plt.legend(loc=0)
```
Exercise 4
```
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

def alg1(data):
  data = list(data)
  changes = True
  while changes:
    changes = False
    for i in range(len(data) - 1):
      if data[i + 1] < data[i]:
        data[i], data[i + 1] = data[i + 1], data[i]
        changes = True
  return data

def alg2(data):
  if len(data) <= 1:
    return data
  else:
    split = len(data) // 2
    left = iter(alg2(data[:split]))
    right = iter(alg2(data[split:]))
    result = []
    # note: this takes the top items off the left and right piles
    left_top = next(left)
    right_top = next(right)
    while True:
      if left_top < right_top:
        result.append(left_top)
        try:
          left_top = next(left)
        except StopIteration:
          # nothing remains on the left; add the right + return
          return result + [right_top] + list(right)
      else:
        result.append(right_top)
        try:
          right_top = next(right)
        except StopIteration:
          # nothing remains on the right; add the left + return
          return result + [left_top] + list(left)

def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

def data2(n):
    return list(range(n))

def data3(n):
    return list(range(n, 0, -1))


log_n = list(np.logspace(0, 4, dtype = int))
alg1data1_time = []
alg1data2_time = []
alg1data3_time = []
alg2data1_time = []
alg2data2_time = []
alg2data3_time = []

for n in log_n:

  data1_list = data1(n)
  data2_list = data2(n)
  data3_list = data3(n)

  alg1data1_start = perf_counter()
  alg1(data1_list)
  alg1data1_stop = perf_counter()
  alg1data1_time.append(alg1data1_stop - alg1data1_start)

  alg1data2_start = perf_counter()
  alg1(data2_list)
  alg1data2_stop = perf_counter()
  alg1data2_time.append(alg1data2_stop - alg1data2_start)

  alg1data3_start = perf_counter()
  alg1(data3_list)
  alg1data3_stop = perf_counter()
  alg1data3_time.append(alg1data3_stop - alg1data3_start)

  alg2data1_start = perf_counter()
  alg2(data1_list)
  alg2data1_stop = perf_counter()
  alg2data1_time.append(alg2data1_stop - alg2data1_start)

  alg2data2_start = perf_counter()
  alg2(data2_list)
  alg2data2_stop = perf_counter()
  alg2data2_time.append(alg2data2_stop - alg2data2_start)

  alg2data3_start = perf_counter()
  alg2(data3_list)
  alg2data3_stop = perf_counter()
  alg2data3_time.append(alg2data3_stop - alg2data3_start)

plt.loglog(log_n, alg1data1_time, label = 'alg1 data1')
plt.loglog(log_n, alg1data2_time, label = 'alg1 data2')
plt.loglog(log_n, alg1data3_time, label = 'alg1 data3')
plt.loglog(log_n, alg2data1_time, label = 'alg2 data1')
plt.loglog(log_n, alg2data2_time, label = 'alg2 data2')
plt.loglog(log_n, alg2data3_time, label = 'alg2 data3')
plt.xlabel('Length')
plt.ylabel('time')

plt.legend(loc=0)
```
```
plt.title('data1 plot', fontsize=15)
plt.loglog(log_n, alg1data1_time, label = 'alg1')
plt.loglog(log_n, alg2data1_time, label = 'alg2')
plt.xlabel('Length')
plt.ylabel('time')
plt.legend(loc=0

plt.title('data2 plot', fontsize=15)
plt.loglog(log_n, alg1data2_time, label = 'alg1')
plt.loglog(log_n, alg2data2_time, label = 'alg2')
plt.xlabel('Length')
plt.ylabel('time')
plt.legend(loc=0)

plt.title('data3 plot', fontsize=15)
plt.loglog(log_n, alg1data3_time, label = 'alg1')
plt.loglog(log_n, alg2data3_time, label = 'alg2')
plt.xlabel('Length')
plt.ylabel('time')
plt.legend(loc=0)

plt.loglog(title = 'alg1 plot')
plt.loglog(log_n, alg1data1_time, label = 'data1')
plt.loglog(log_n, alg1data2_time, label = 'data2')
plt.legend(loc=0)

```

Parallel:
```
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Pool, Manager

log_n = list(np.logspace(0, 7, dtype = int))

def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

def alg2(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(alg2(data[:split])) # left sorted half
        right = iter(alg2(data[split:])) # right sorted half
        result = []
        # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
        while True:
            if left_top < right_top:
                result.append(left_top)
            try:
                left_top = next(left)
            except StopIteration:
                # nothing remains on the left; add the right + return
                return result + [right_top] + list(right)
            else:
                result.append(right_top)
            try:
                right_top = next(right)
            except StopIteration:
                # nothing remains on the right; add the left + return
                return result + [left_top] + list(left)

def merge(left_h, right_h):
    if len(right_h) <= 1:
        return right_h
    else:
        left = iter(left_h)
        right = iter(right_h)
        result = []
        # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
        while True:
            if left_top < right_top:
                result.append(left_top)
                try:
                    left_top = next(left)
                except StopIteration:
                # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right)
            else:
                result.append(right_top)
                try:
                    right_top = next(right)
                except StopIteration:
                # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left)

def main():
    total_time = []
    with Pool(2) as workers:
        for n in log_n:
            data1_list = data1(n)
            start_time = perf_counter()
            
            split = len(data1_list) // 2
            left_half = data1_list[:split]
            right_half = data1_list[split:]
            left_sorted, right_sorted = workers.map(alg2, [left_half, right_half])
            
            merge(left_sorted, right_sorted)

            stop_time = perf_counter()
            total_time.append(stop_time - start_time)

    plt.loglog(log_n, total_time, label = 'alg2 data1')
    plt.xlabel('Length')
    plt.ylabel('time')
    plt.show()

    plt.legend(loc=0)

if __name__ == '__main__':
    main()
```
