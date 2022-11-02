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

alzh_pmids_list = get_pmids("Alzheimers", 2022)
cancer_pmids_list = get_pmids("Cancer", 2022)

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
Calculate the Alzheimers dataset's SPECTER
```
alzh_specter = specter(alzh_metadata_dict)
```
Calculate the Alzheimers dataset's SPECTER
```
alzh_specter = specter(alzh_metadata_dict)
```


### Exercise 4
The dataset I chose is an IT Career Proficiency Dataset from Mendeley Data. There are around 9000 survey responses and 18 variables. The data was from a survey for professionsls who assessed their abilities in various IT and software fields. The key variables are specified. None of the variables are redundant or exactly derivable. Ideas for predictions include predicting tech proficiency based on job ("Role") or  predicting role based on interest or level of qualification in tech fields. As all of the variables are related to level of expertise in IT fields, most of the variables can be trained and predicted with models. The data is in a standard format.

The dataset was published on 28 Oct 2022 and is licensed under a Creative Commons Attribution 4.0 International license. The data can be shared, copied, and modified as long as due credit is given, a link to the CC BY license is given, and any changes made are indicated. Diagnostic and prescriptive analyses cannot be done on this dataset but Predictive analyses, such as job prediction previously mentioned, and Descriptive Analyses, such as clustering and classification, are suitable for this dataset.
The link to the data is: https://data.mendeley.com/datasets/kzt6h7pz97/1
