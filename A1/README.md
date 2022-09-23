Goldie Zhu
Assignment 1 README

This assignment was written on Colab

Exercise 1:
To test more cases, add them with the human and chicken testers. This works with any testers so you can add more testers (e.g. dog_tester) and it will work.

Exercise 2:
There are 152361 rows and 4 columns. The four column categories are name, age, weight, and eyecolor. There are 152360 people.

The summary of the dataset and the distribution information is in the code. The distribution of ages is relatively uniform until the reported age is above 60 years old. This is expected because people have varying lifespans and there is a dramatic dropoff in the amount of people with longer lifespans. For weight, the weights below 60 to 80 kilograms tended to be measurements of the younger population. Unless the person was suffering from a health issue, the weight measurements tended to be consistently increasing. The outliers were for people who were over- or under-weight for their age.

The general relationship between age and weight was consistently increasing. This means that during the growth phase, children and young adults tended to consistently increase in weight with age. By the age of 25, there is little consistency in weight and there is no longer a correlation between weight and age. There is a lot of variance in weight after the age of 25. The outlier, Anthony Freeman, was identified by limiting the parameters by age and weight. 

Exercise 3:
The historical data for COVID-19 cases by state was downloaded from the New York Time's Github on September 21, 2022.

For the function using a list of state names, you can change the states in the list by changing the state_names[] function. There aren't any specific limitations other than the amount of data, which could overcrowd the graph, and having to change the states in the code itself instead of taking an input. Only valid state names with the proper spelling and upper case letters are accepted. Examples of plots in use is any kind of comparison plot; this could be comparing organizations, companies, states, countries, and other separate entities. This could be used to study the allocation of resources, trending behaviors, effective policies, and other necessary comparisons.

For the function comparing two states, it takes two inputs that are separated by a comma and finds which one had the highest number of daily new cases. It then finds how many days passed between the peaks of the two states. This can be used to look for trends between diseases in different places, as shown in the code example, but it could also be reversed to look for the lowest amount for something of your choosing. This could be used to analyze geographical trends, policy effectiveness, and more.

Exercise 4:
The DescriptorName associated with DescriptorUI D007154 is "Immune Ststem Diseases."

