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

PC1 vs PC2 graphed together shows a lot of overlap and not a lot of separation. This is not desirable and means that the data differences between the variables of the two groups are small. PC0 vs PC2 graphed together shows a clear separation and two prominent clusters slanting towards each other while PC0 vs PC2 graphed together shows a clear separation and two prominent clusters. This indicates that there is a clear difference in the variables of the two groups as similar observations are put together in clusters. The possible slight overlap means that the data from the two groups are slightly similar but still very different as the clusters are clearly separated.


### Exercise 4
The dataset I chose is an IT Career Proficiency Dataset from Mendeley Data. There are around 9000 (9179 lines) survey responses and 18 variables. The data was from a survey for professionals who assessed their abilities in various IT and software fields. The key variables are specified. None of the variables are redundant or exactly derivable. Ideas for predictions include predicting tech proficiency based on job ("Role") or  predicting role based on interest or level of qualification in tech fields. As all of the variables are related to level of expertise in IT fields, most of the variables can be trained and predicted with models. The data is in a standard format.

The dataset was published on 28 Oct 2022 and is licensed under a Creative Commons Attribution 4.0 International license. The data can be shared, copied, and modified as long as due credit is given, a link to the CC BY license is given, and any changes made are indicated. Diagnostic and prescriptive analyses cannot be done on this dataset but Predictive analyses, such as job prediction previously mentioned, and Descriptive Analyses, such as clustering and classification, are suitable for this dataset.
The link to the data is: https://data.mendeley.com/datasets/kzt6h7pz97/1

My data does not need to be cleaned because I ran code to check for null/NaN values and ambiguous data points, which there were none. This is done with isnull() and df.(column name).unique().

This data set is likely not legitimate as the analysis of it showed suspiciously similar numbers. The distribution of the numbers are the same across all "Roles" and even the roles with different column values showed the same rounded values. 

Below is some analysis and charts that were done:
<img width="465" alt="Screen Shot 2022-11-02 at 1 36 02 PM" src="https://user-images.githubusercontent.com/37753494/199562127-db838b5c-36cc-43f3-9795-1dc0087d6089.png">
This dataframe extends right and shows that there are no null values.

<img width="870" alt="Screen Shot 2022-11-02 at 1 39 46 PM" src="https://user-images.githubusercontent.com/37753494/199562198-7f8c7b08-6bdf-4bb9-b3ec-7dcd9e26f3a7.png">
df1 is for database administrator roles while df2 is for hardware engineers. It is strange that the distribution of values is the same for different roles.

<img width="374" alt="Screen Shot 2022-11-02 at 1 41 02 PM" src="https://user-images.githubusercontent.com/37753494/199562432-ea9756bb-67e3-491e-9a36-0ef0e54ea82f.png">
This is the distribution of values for different columns on the full dataset. This analysis was done on ten columns and returned the same distribution for all of them.

<img width="361" alt="Screen Shot 2022-11-02 at 1 44 36 PM" src="https://user-<img width="468" alt="Screen Shot 2022-11-02 at 1 45 44 PM" src="https://user-images.githubusercontent.com/37753494/199563423-421c2388-97c0-4b77-9284-497230778958.png">
<img width="463" alt="Screen Shot 2022-11-02 at 1 45 56 PM" src="https://user-images.githubusercontent.com/37753494/199563427-f52e634e-61f7-4553-980c-03c273fcb12a.png">
<img width="465" alt="Screen Shot 2022-11-02 at 1 45 51 PM" src="https://user-images.githubusercontent.com/37753494/199563428-34fe5cd2-47c9-4bfe-806e-4c0e409b70a6.png">
images.githubusercontent.com/37753494/199563306-9c47a7c9-3f19-4b3e-811a-9a765d312848.png">

