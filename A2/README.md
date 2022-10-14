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
This list is compared with another list 'alphabet_list', which contains every character of the alphabet. With two 'for' loops, each character in the inputted word is switched with every letter in the alphabet. After each change, the number is put through the hashes and the word is checked to see if it is in the bloom filter, which already had words inputted in the implementation.
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

3. 

### Exercise 3
### Exercise 4
