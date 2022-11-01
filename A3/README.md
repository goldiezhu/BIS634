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

###Exercise 4
The dataset I chose is an IT Career Proficiency Dataset from Mendeley Data. There are around 9000 survey responses and 18 variables. The data was from a survey for professionsls who assessed their abilities in various IT and software fields. The key variables are specified. None of the variables are redundant or exactly derivable. Ideas for predictions include predicting tech proficiency based on job ("Role") or  predicting role based on interest or level of qualification in tech fields. As all of the variables are related to level of expertise in IT fields, most of the variables can be trained and predicted with models. The data is in a standard format.

The dataset was published on 28 Oct 2022 and is licensed under a Creative Commons Attribution 4.0 International license. The data can be shared, copied, and modified as long as due credit is given, a link to the CC BY license is given, and any changes made are indicated. Diagnostic and prescriptive analyses cannot be done on this dataset but Predictive analyses, such as job prediction previously mentioned, and Descriptive Analyses, such as clustering and classification, are suitable for this dataset.
The link to the data is: https://data.mendeley.com/datasets/kzt6h7pz97/1
