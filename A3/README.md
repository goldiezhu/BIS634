## BIS634 Assignment 3
### Goldie Zhu

### Exercise 1
Function to identify PubMed IDs
```
def get_pmids(condition, year):
    pmids = []
    r = requests.get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={condition}+AND+{year}[pdat]&retmode=xml&retmax=1000"
    )
    doc = m.parseString(r.text)
    ids = doc.getElementsByTagName("Id")
    for i in ids:
        pmids.append(i.childNodes[0].wholeText)
        #print(i.childNodes[0].wholeText)
    return pmids

#get PMIDS
alzh_pmids_list = get_pmids("Alzheimers", 2022)
cancer_pmids_list = get_pmids("Cancer", 2022)

#Change list to comma separated list
alzh_pmids = ",".join(alzh_pmids_list)
cancer_pmids = ",".join(cancer_pmids_list)
```
Function to pull metadata and store in JSON
```
def get_info(pmids):
    metadata_dict_w_pmid = {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    r = requests.post(
        url, data = {
                        "id": pmids,
                        "db": "PubMed",
                        "retmode": "xml"
                    }
    )

    doc = m.parseString(r.text)
    #print(pmid_current_decoded)
    
    articles = doc.getElementsByTagName("MedlineCitation")
    #print("articles.length", articles.length)
    counter = 0
    for i in articles:
        metadata_dict = {}
        
        #stores PMID
        pmid_current_tag = doc.getElementsByTagName("PMID")
        pmid_current = ET.fromstring(pmid_current_tag[counter].toxml())
        pmid_current_decoded = ET.tostring(pmid_current, method = "text").decode()
        #print(pmid_current_decoded)
        
        all_title = ""
        all_abstract = ""
        titles = i.getElementsByTagName("ArticleTitle")
        abstracts = i.getElementsByTagName("AbstractText")
        #Stores Title(s)
        try:
            title.length >= 1
        except:
            for j in range(titles.length):
                title_forDecode = ET.fromstring(titles[j].toxml())
                title = ET.tostring(title_forDecode, method = "text").decode()
                all_title += title + " "   
        else:
            all_title = "No title"
        
        #Stores Abstracts
        try:
            abstracts.length >= 1
        except:
            all_abstract = "No Abstract"
        else:
            for k in range(abstracts.length):
                abstract_forDecode = ET.fromstring(abstracts[k].toxml())
                abstract = ET.tostring(abstract_forDecode, method = "text").decode()
                all_abstract += abstract + " "
        
        #Stores Query
        if pmid_current_decoded in alzh_pmids:
            query = "Alzheimers"
        else:
            query = "Cancer"

        #Insert into dictionary
        metadata_dict["ArticleTitle"] = all_title
        metadata_dict["AbstractText"] = all_abstract
        metadata_dict["query"] = query
        metadata_dict_w_pmid[pmid_current_decoded] = metadata_dict
        counter += 1

    #print("counter", counter)
    return metadata_dict_w_pmid

#Get metadata in the form of a dictionary
alzh_metadata_dict = get_info(alzh_pmids)
cancer_metadata_dict = get_info(cancer_pmids)

#Dump dictionary into a JSON
alzh_dump = json.dumps(alzh_metadata_dict, indent=4)
cancer_dump = json.dumps(cancer_metadata_dict, indent=4)
 
with open("alzheimer_metadata.json", "w") as outfile:
    outfile.write(alzh_dump)
with open("cancer_metadata.json", "w") as outfile:
    outfile.write(cancer_dump)
```

The overlap in the two sets has a PMID of ['36314209'].

For papers with multiple AbstractText fields, I concatenated all of the different parts of the abstract. It leaves on AbstractText without multiple parts or spaces between the different parts but this could be make the AbstractText more difficult or less clear to read to the naked eye.


### Exercise 2
Function to calculate SPECTER
```
def specter(papers):
    embeddings = {}
    for pmid, paper in tqdm.tqdm(papers.items()):
        data = [paper["ArticleTitle"] + tokenizer.sep_token + paper["AbstractText"]]
        inputs = tokenizer(
            data, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        result = model(**inputs)
        # take the first token in the batch as the embedding
        embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]

    # turn our dictionary into a list
    embeddings = [embeddings[pmid] for pmid in papers.keys()]
    return embeddings
```
Combine Dictionaries and Calculate the SPECTER
```
z = {**alzh_metadata_dict, **cancer_metadata_dict}
combined_specter = specter(z)
```
Function to calculate PCA
```
from sklearn import decomposition
import pandas as pd

def pca(embeddings, papers):
    pca = decomposition.PCA(n_components=3)
    embeddings_pca = pd.DataFrame(
        pca.fit_transform(embeddings),
        columns=['PC0', 'PC1', 'PC2']
    )
    embeddings_pca["query"] = [paper["query"] for paper in papers.values()]
    return embeddings_pca
```
Calculate PCAs
```
combined_pca = pca(combined_specter, z)
print(combined_pca)
```

Output

<img width="358" alt="Screen Shot 2022-11-01 at 9 15 31 PM" src="https://user-images.githubusercontent.com/37753494/199371838-c8293f12-3f7c-4609-94d9-55113f01a4f6.png">


Graph PCAs
```
import seaborn as sns

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="PC0", y="PC1",
    hue="query",
    palette=sns.color_palette("deep", 2),
    data=result,
    legend="full",
    alpha=0.3
)
```

PCA graphs

<img width="894" alt="Screen Shot 2022-11-01 at 9 14 10 PM" src="https://user-images.githubusercontent.com/37753494/199371767-dd569439-84e9-4c5a-8191-3a2f42aa114b.png">

<img width="886" alt="Screen Shot 2022-11-01 at 9 14 18 PM" src="https://user-images.githubusercontent.com/37753494/199371779-0b2baf86-b2ef-4e6d-b49a-99e22425554e.png">


<img width="890" alt="Screen Shot 2022-11-01 at 9 14 29 PM" src="https://user-images.githubusercontent.com/37753494/199371794-c24e4534-3d58-45cb-a3b5-3f5ec4c0052d.png">

PC1 vs PC2 graphed together shows a lot of overlap and not a lot of separation. This is not desirable and means that the data differences between the variables of the two groups are small. PC0 vs PC2 graphed together shows a clear separation and two prominent clusters slanting towards each other while PC0 vs PC2 graphed together shows a clear separation and two prominent clusters. This indicates that there is a clear difference in the variables of the two groups as similar observations are put together in clusters. The possible slight overlap means that the data from the two groups are slightly similar but still very different as the clusters are clearly separated. When they are separated, they are more significant.

### Exercise 3
Function for Explicit Euler method
```
def explicitEuler(s, i, r, beta, gamma, N, timestep):
    dsdt = -beta/N * s * i
    didt = beta/N * s * i - gamma * i
    drdt = gamma * i
    s_updated = s + dsdt * timestep
    i_updated = i + didt * timestep
    r_updated = r + drdt * timestep
    return s_updated, i_updated, r_updated
```
SIR method
```
s_values = []
i_values = []
r_values = []

def SIR(beta, gamma, timestep, Tmax, infected, N, s, i, r):
    peak_day = 0
    sick_prev = 1
    s_values.append(N - infected)
    i_values.append(infected)
    r_values.append(0)

    for j in range(Tmax):
        s_updated, i_updated, r_updated = explicitEuler(s_values[j], i_values[j], r_values[j], beta, gamma, N, timestep)
        s_values.append(s_updated)
        i_values.append(i_updated)
        r_values.append(r_updated)

        if i_updated > sick_prev:
            sick_prev = i_updated
            peak_day = j
            
    print("The disease cases peaks on day", peak_day + 1, "at a case number of", sick_prev, "people")
    
    return s_values, i_values, r_values, peak_day, sick_prev
```
Plot Euler Method
```
import matplotlib.pyplot as plt

SIR(2, 1, 1, 25, 1, 134000, 133999, 1, 0)
plt.figure()
plt.title("Euler method")
plt.plot(s_values, color = 'orange', label='Susceptible')
plt.plot(i_values, 'r', label='Infected')
plt.plot(r_values, 'g', label='Recovered with immunity')
plt.grid()
plt.xlabel("timestep, $t$ [s]")
plt.ylabel("Numbers of individuals")
plt.legend(loc = "best")
plt.show()
```
Time course of infection

<img width="597" alt="Screen Shot 2022-11-03 at 12 07 31 AM" src="https://user-images.githubusercontent.com/37753494/199645774-30823b10-19f9-4e10-bced-e26f5896de43.png">

The number of infected people drops below 1 on day 21. The number of infected people peaked at 26034 on day 16.

Heat map values
```
x = 2
y = 1
day_peak = np.empty((20, 20), int)
indiv_peak = np.empty((20, 20), int)

for i in range(20):
    for j in range(20):
        a, b, c, d, e = SIR(i, j, 1, 20, 1, 134000, 133999, 1, 0)
        day_peak[i][j] = d + 1
        indiv_peak[i][j] = e
```
Peak day heat map
<img width="489" alt="Screen Shot 2022-11-03 at 12 09 52 AM" src="https://user-images.githubusercontent.com/37753494/199646003-873e0327-5ff9-4ea3-bb29-2ec5e5369e1a.png">
Peak number of individuals heat map
<img width="515" alt="Screen Shot 2022-11-03 at 12 10 16 AM" src="https://user-images.githubusercontent.com/37753494/199646044-d79ba7bc-ac13-4885-873f-9ce4da320979.png">


### Exercise 4
The dataset I chose is an IT Career Proficiency Dataset from Mendeley Data. There are around 9000 (9179 lines) survey responses and 18 variables. The data was from a survey for professionals who assessed their abilities in various IT and software fields. The key variables are specified. None of the variables are redundant or exactly derivable. Ideas for predictions include predicting tech proficiency based on job ("Role") or  predicting role based on interest or level of qualification in tech fields. As all of the variables are related to level of expertise in IT fields, most of the variables can be trained and predicted with models. The data is in a standard format.

The dataset was published on 28 Oct 2022 and is licensed under a Creative Commons Attribution 4.0 International license. The data can be shared, copied, and modified as long as due credit is given, a link to the CC BY license is given, and any changes made are indicated. Diagnostic and prescriptive analyses cannot be done on this dataset but Predictive analyses, such as job prediction previously mentioned, and Descriptive Analyses, such as clustering and classification, are suitable for this dataset.
The link to the data is: https://data.mendeley.com/datasets/kzt6h7pz97/1

My data does not need to be cleaned because I ran code to check for null/NaN values and ambiguous data points, which there were none. This is done with isnull() and df.(column name).unique().

This data set is likely not legitimate as the analysis of it showed suspiciously similar numbers. The distribution of the numbers are the same across all "Roles" and even the roles with different column values showed the same rounded values. 

Below is some analysis and charts that were done (More tests can be seen in the code but they show the same information):

<img width="465" alt="Screen Shot 2022-11-02 at 1 36 02 PM" src="https://user-images.githubusercontent.com/37753494/199562127-db838b5c-36cc-43f3-9795-1dc0087d6089.png">
This dataframe extends right and shows that there are no null values.

<img width="870" alt="Screen Shot 2022-11-02 at 1 39 46 PM" src="https://user-images.githubusercontent.com/37753494/199562198-7f8c7b08-6bdf-4bb9-b3ec-7dcd9e26f3a7.png">
df1 is for database administrator roles while df2 is for hardware engineers. It is strange that the distribution of values is the same for different roles.

<img width="374" alt="Screen Shot 2022-11-02 at 1 41 02 PM" src="https://user-images.githubusercontent.com/37753494/199562432-ea9756bb-67e3-491e-9a36-0ef0e54ea82f.png">
This is the distribution of values for different columns on the full dataset. This analysis was done on ten columns and returned the same distribution for all of them.

<img width="361" alt="Screen Shot 2022-11-02 at 1 44 36 PM" src="https://user-images.githubusercontent.com/37753494/199563886-4eb8f3e0-68da-4a3c-b52b-222cbd78aad8.png">
<img width="468" alt="Screen Shot 2022-11-02 at 1 45 44 PM" src="https://user-images.githubusercontent.com/37753494/199563423-421c2388-97c0-4b77-9284-497230778958.png">
<img width="463" alt="Screen Shot 2022-11-02 at 1 45 56 PM" src="https://user-images.githubusercontent.com/37753494/199563427-f52e634e-61f7-4553-980c-03c273fcb12a.png">
<img width="465" alt="Screen Shot 2022-11-02 at 1 45 51 PM" src="https://user-images.githubusercontent.com/37753494/199563428-34fe5cd2-47c9-4bfe-806e-4c0e409b70a6.png">


Appendix:

Exercise 1:
```
from inspect import isdatadescriptor
import requests
import xml.dom.minidom as m
import xml.etree.ElementTree as ET
import json

def get_pmids(condition, year):
    pmids = []
    r = requests.get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={condition}+AND+{year}[pdat]&retmode=xml&retmax=1000"
    )
    doc = m.parseString(r.text)
    ids = doc.getElementsByTagName("Id")
    for i in ids:
        pmids.append(i.childNodes[0].wholeText)
        #print(i.childNodes[0].wholeText)
    return pmids

alzh_pmids_list = get_pmids("Alzheimers", 2022)
cancer_pmids_list = get_pmids("Cancer", 2022)

alzh_pmids = ",".join(alzh_pmids_list)
cancer_pmids = ",".join(cancer_pmids_list)

#print(alzh_pmids, "\n", cancer_pmids)

def get_info(pmids):
    metadata_dict_w_pmid = {}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    r = requests.post(
        url, data = {
                        "id": pmids,
                        "db": "PubMed",
                        "retmode": "xml"
                    }
    )

    doc = m.parseString(r.text)
    #print(pmid_current_decoded)
    
    articles = doc.getElementsByTagName("MedlineCitation")
    #print("articles.length", articles.length)
    counter = 0
    for i in articles:
        metadata_dict = {}

        pmid_current_tag = doc.getElementsByTagName("PMID")
        pmid_current = ET.fromstring(pmid_current_tag[counter].toxml())
        pmid_current_decoded = ET.tostring(pmid_current, method = "text").decode()
        #print(pmid_current_decoded)
        
        all_title = ""
        all_abstract = ""
        titles = i.getElementsByTagName("ArticleTitle")
        abstracts = i.getElementsByTagName("AbstractText")
        try:
            title.length >= 1
        except:
            for j in range(titles.length):
                title_forDecode = ET.fromstring(titles[j].toxml())
                title = ET.tostring(title_forDecode, method = "text").decode()
                all_title += title + " "   
        else:
            all_title = "No title"
            

        try:
            abstracts.length >= 1
        except:
            all_abstract = "No Abstract"
        else:
            for k in range(abstracts.length):
                abstract_forDecode = ET.fromstring(abstracts[k].toxml())
                abstract = ET.tostring(abstract_forDecode, method = "text").decode()
                all_abstract += abstract + " "

        if pmid_current_decoded in alzh_pmids:
            query = "Alzheimers"
        else:
            query = "Cancer"

        metadata_dict["ArticleTitle"] = all_title
        metadata_dict["AbstractText"] = all_abstract
        metadata_dict["query"] = query
        metadata_dict_w_pmid[pmid_current_decoded] = metadata_dict
        counter += 1

    #print("counter", counter)
    return metadata_dict_w_pmid

alzh_metadata_dict = get_info(alzh_pmids)
cancer_metadata_dict = get_info(cancer_pmids)

alzh_dump = json.dumps(alzh_metadata_dict, indent=4)
cancer_dump = json.dumps(cancer_metadata_dict, indent=4)
 
with open("alzheimer_metadata.json", "w") as outfile:
    outfile.write(alzh_dump)
with open("cancer_metadata.json", "w") as outfile:
    outfile.write(cancer_dump)
    
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

print(intersection(alzh_pmids_list,cancer_pmids_list))
```

Exercise 2:
```
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

import tqdm

# we can use a persistent dictionary (via shelve) so we can stop and restart if needed
# alternatively, do the same but with embeddings starting as an empty dictionary

def specter(papers):
    embeddings = {}
    for pmid, paper in tqdm.tqdm(papers.items()):
        data = [paper["ArticleTitle"] + tokenizer.sep_token + paper["AbstractText"]]
        inputs = tokenizer(
            data, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        result = model(**inputs)
        # take the first token in the batch as the embedding
        embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]

    # turn our dictionary into a list
    embeddings = [embeddings[pmid] for pmid in papers.keys()]
    return embeddings
    
z = {**alzh_metadata_dict, **cancer_metadata_dict}
combined_specter = specter(z)

from sklearn import decomposition
import pandas as pd

def pca(embeddings, papers):
    pca = decomposition.PCA(n_components=3)
    embeddings_pca = pd.DataFrame(
        pca.fit_transform(embeddings),
        columns=['PC0', 'PC1', 'PC2']
    )
    embeddings_pca["query"] = [paper["query"] for paper in papers.values()]
    return embeddings_pca

combined_pca = pca(combined_specter, z)
print(combined_pca)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="PC0", y="PC1",
    hue="query",
    palette=sns.color_palette("deep", 2),
    data=combined_pca,
    legend="full",
    alpha=0.3
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="PC0", y="PC2",
    hue="query",
    palette=sns.color_palette("deep", 2),
    data=combined_pca,
    legend="full",
    alpha=0.3
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="PC1", y="PC2",
    hue="query",
    palette=sns.color_palette("deep", 2),
    data=combined_pca,
    legend="full",
    alpha=0.3
)
```
Exercise 3:
```
import numpy as np
def explicitEuler(s, i, r, beta, gamma, N, timestep):
    dsdt = -beta/N * s * i
    didt = beta/N * s * i - gamma * i
    drdt = gamma * i
    s_updated = s + dsdt * timestep
    i_updated = i + didt * timestep
    r_updated = r + drdt * timestep
    return s_updated, i_updated, r_updated
    
s_values = []
i_values = []
r_values = []

def SIR(beta, gamma, timestep, Tmax, infected, N, s, i, r):
    peak_day = 0
    sick_prev = 1
    s_values.append(N - infected)
    i_values.append(infected)
    r_values.append(0)

    for j in range(Tmax):
        s_updated, i_updated, r_updated = explicitEuler(s_values[j], i_values[j], r_values[j], beta, gamma, N, timestep)
        s_values.append(s_updated)
        i_values.append(i_updated)
        r_values.append(r_updated)

        if i_updated > sick_prev:
            sick_prev = i_updated
            peak_day = j
            
    print("The disease cases peaks on day", peak_day + 1, "at a case number of", sick_prev, "people")
    
    return s_values, i_values, r_values, peak_day, sick_prev

import matplotlib.pyplot as plt

SIR(2, 1, 1, 25, 1, 134000, 133999, 1, 0)
plt.figure()
plt.title("Euler method")
plt.plot(s_values, color = 'orange', label='Susceptible')
plt.plot(i_values, 'r', label='Infected')
plt.plot(r_values, 'g', label='Recovered with immunity')
plt.grid()
plt.xlabel("timestep, $t$ [s]")
plt.ylabel("Numbers of individuals")
plt.legend(loc = "best")
plt.show()

x = 2
y = 1
day_peak = np.empty((20, 20), int)
indiv_peak = np.empty((20, 20), int)

for i in range(20):
    for j in range(20):
        a, b, c, d, e = SIR(i, j, 1, 20, 1, 134000, 133999, 1, 0)
        day_peak[i][j] = d + 1
        indiv_peak[i][j] = e

import numpy as np
import matplotlib.pyplot as plt

im = plt.imshow(day_peak, cmap=plt.cm.RdBu, extent=(-3, 3, 3, -3), interpolation='bilinear')
plt.colorbar(im)
plt.show()

im = plt.imshow(indiv_peak, cmap=plt.cm.RdBu, extent=(-3, 3, 3, -3), interpolation='bilinear')
plt.colorbar(im)
plt.show()
```


Exercise 4:
``` 
#only includes a few examples as the same code run on different columns returned the same values

import pandas as pd
df = pd.read_csv ('dataset9000.csv')
print(df)

df.info()

df.Role.unique()

df.isnull()

df1 = df[df['Role'] == "Database Administrator"]
print(df1)
df2 = df[df['Role'] == "Hardware Engineer"]
print(df2)

df1['Cyber Security'].value_counts()
df2['Cyber Security'].value_counts()
df2['Computer Architecture'].value_counts()

df['Database Fundamentals'].value_counts()
df2['Distributed Computing Systems'].value_counts()
df['Computer Architecture'].value_counts() 

import numpy as np
import matplotlib.pyplot as plt

df['Distributed Computing Systems'].value_counts().plot(kind='bar', title = "Distributed Computing Systems Competence")
df1['Cyber Security'].value_counts().plot(kind='bar', title = "Computer Architecture Competence for Database Admins")
df2['Computer Architecture'].value_counts().plot(kind='bar', title = "Computer Architecture Competence for Hardware Engineers")
```
