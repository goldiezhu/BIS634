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

### Appendix
Exercise 1
```
```
Exercise 2
```
```
Exercise 3
```
```
Exercise 4
```
```
