# SciSearch - Query-by-Example for Scientific Article Retrieval 
This project is an attempt of implementing and improving on the work of **Sheshera Mysore, Tim O'Gorman, Andrew McCallum, Hamed Zamani** titled ***"CSFCube - A Test Collection of Computer Science Papers for Faceted Query by Example"***

The dataset can be found here: https://github.com/iesl/CSFCube

The paper describing the dataset can be accessed here: https://arxiv.org/abs/2103.12906

Demo video: https://youtu.be/CNITDhWITH0

**Team members:** Ashutosh Kumar Singh, Ashwamegh Rathore, Nakul Aggarwal, Suhas Jain



## Table of Contents ##

<!--ts-->
   * [Aim of the Project](#aim-of-the-project)
   * [Installations](#installations)
   * [Usage Instructions](#usage-instructions)
   * [Dataset Description](#dataset-description)
   * [Directory Structure](#directory-structure)
        * [Data](#data)
        * [Models-KLDivLoss](#models-kldivloss)
        * [Models-NLLLoss](#models-nllloss)
        * [Plots-KLDivLoss](#plots-kldivloss)
        * [Plots-NLLoss](#plots-nllloss)
        * [GUI](#gui)
        * [Preprocess](#preprocess)
            * [parse.py](#parse.py)
            * [embeddings.ipynb](#embeddings.ipynb)
        * [requirements.txt](#requirements.txt)
        * [QBE-KLDivLoss.ipynb](#qbe-kldivloss.ipynb)
        * [QBE-NLLLoss.ipynb](#qbe-nllloss.ipynb)
   * [Results](#results)


<!--te-->


## Aim of the Project ##

In research, it is naturally difficult to express in-
formation requirements as simple keyword queries
because they can have a very broad meaning and
we might not be able to retrieve a specific section
from the whole bundle of documents. A far more
effective way to understand the needs of the user is
by asking them to give a research paper deemed as
relevant to their aspirations. Hence, we aim to envsion and develop a model that is able to retrieve scientific papers analogous to a query scientific paper,
along specifically chosen rhetorical structure elements (facets/aspects), like background/objective,
method and results. So we frame our task as one of
retrieving scientific papers given a query paper and
additional information indicating the query facet.
We want to give any researcher the freedom to ask – *“I came across the paper XYZ during my research
and am extremely inclined towards the results and
mathematical background given by the author. Can
you please get me some more papers that I might
find relevant to my research?”*


## Installations  ##


The `requirements.txt` file should list all Python libraries that your notebooks
depend on, and they will be installed using:

```
pip install -r requirements.txt
```

## Usage Instructions ##

### Training Neural Network ###
---------------
Open the file `QBE-KLDivLoss.ipynb/QBE-NLLLoss.ipynb` in jupyter-notebook/vscode with jupyter-npotebook extension and run all cells to train the Neural Network.

The training weights will be stored in the `models-NLLLoss/` folder.

This will **take time** so we recommend skipping the training step and moving ahead to testing the result of our Neural Network.

### Streamlit App ###
---------------
The app has been deployed [here](https://share.streamlit.io/suhas4122/scisearch/main/SciSearch.py).

If for any reason the above link does not work, please find a copy of the model also hosted [here](https://share.streamlit.io/ashutoshaks/scisearch/main/SciSearch.py)

The user may choose the facet from the options-
Background, method and result.

The user may choose the encoding scheme and also the loss function for training the model

Finally, enter the search query and the number of documents to be retrieved.

You will find the list of ranked documents in the Query Result section.

**Note:** 
If then above 2 links fail to open kindly install streamlit using the command (you might have already installed it while installing requirements.txt)-
```bash
pip install streamlit
```
To test installation, use-
```bash
streamlit hello
```
To run streamlit app enter the command given below-
```bash
cd gui/
streamlit run SciSearch.py
```
If you are having trouble in installing pip please refer this website: https://docs.streamlit.io/library/get-started/installation
## Dataset Description ##

```bash
├── data/abstracts-csfcube-preds-no-unicode.jsonl
├── data/abstracts-csfcube-preds.json
├── data/abstracts-csfcube-preds.jsonl
├── data/evaluation_splits.json
├── data/test-pid2anns-csfcube-background.json
├── data/test-pid2anns-csfcube-method.json
├── data/test-pid2anns-csfcube-result.json
├── data/test-pid2pool-csfcube.json
```

`abstracts-csfcube-preds.{jsonl/json}`: jsonl/json file containing the paper-id, abstracts, titles, and metadata for the queries and candidates which are part of the test collection.

`abstracts-csfcube-preds.jsonl`: jsonl file containing the same content as `abstracts-csfcube-preds.{jsonl/json}` minus the unicode characters.


`test-pid2anns-csfcube-{background/method/result}.json`: JSON file with the query paper-id, candidate paper-ids for every query paper in the test collection. Use these files in conjunction with `abstracts-csfcube-preds.jsonl` to generate files for use in model evaluation. 

`test-pid2pool-csfcube.json`: JSON file query paper-id, candidate paper-ids and the methods which caused the candidate to be included in the pool. The methods are one among {abs_tfidf, abs_cbow200, abs_tfidfcbow200, title_tfidf, title_cbow200, title_tfidfcbow200, specter, cited}. This file is included to facilitate further analysis of the dataset.

`evaluation_splits.json`: Paper-ids for the splits to use in reporting evaluation numbers. `eval_scripts` implements the evaluation protocol and computes evaluation metrics. 



## Directory Structure  ##

```bash
.
├── LICENSE
├── QBE-KLDivLoss.ipynb
├── QBE-NLLLoss.ipynb
├── README.md
├── __pycache__
├── data
├── gui
├── models-KLDivLoss
├── models-NLLLoss
├── plots-KLDivLoss
├── plots-NLLLoss
├── preprocess
└── requirements.txt
```
The structure and purpose of each and every file/directory is cleary mentioned below-
### Data ###
---------------
```bash 
data
├── abstracts-csfcube-preds-no-unicode.jsonl
├── abstracts-csfcube-preds.json
├── abstracts-csfcube-preds.jsonl
├── bert_nli
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   └── result.json
├── bert_pp
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   └── result.json
├── evaluation_splits.json
├── scibert_cased
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   └── result.json
├── scibert_uncased
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   └── result.json
├── specter
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   └── result.json
├── susimcse
│   ├── all.json
│   ├── background.json
│   ├── method.json
│   └── result.json
├── test-pid2anns-csfcube-background.json
├── test-pid2anns-csfcube-method.json
├── test-pid2anns-csfcube-result.json
├── test-pid2pool-csfcube.json
└── unsimcse
    ├── all.json
    ├── background.json
    ├── method.json
    └── result.json

```
This folder contains the [dataset](#dataset-description) as discussed above. It also contains the embeddings generated in the precrocessing step.
### Models KLDivLoss ###
---------------
This folder contains the weights of the Neural Network after training using KLDivLoss.
### Models NLLLoss ###
---------------
This folder contains the weights of the Neural Network after training using NLLLoss.
### Plots KLDivLoss ###
---------------
This folder contains the results measured by metrics such as-accuracy and loss for various facets using different encoding scheme when trained using KLDivLoss.
### Plots NLLLoss ###
---------------
This folder contains the results measured by metrics such as-accuracy and loss for various facets using different encoding scheme when trained using NLLLoss.

### gui ###
---------------

 ```bash
gui
├── SciSearch.py
└── gui_helper.py
```
This folder contains code for creating a front-end portal for our model. We built and hosted this portal using [Streamlit](https://streamlit.io/).



### preprocess ###
---------------
```bash
preprocess
├── embeddings.ipynb
└── parse.py
```

### parse.py ###
---------------
This file contains code for parsing the dataset to get individual sentences belonging to the classes- background,method and result.
### embeddings.ipynb ###
---------------
**NOTE:** It is **not recommended** that you run the embeddings.ipynb file as it will download GBs of data and take a lot of time to run. The embeddings have already been calculated and stored in `data/` folder, so further files will run without any issues.

This file contains code for creating embeddings using pre-trained models such as-

- Unsupervised Simple Contrastive Learning of Sentence Embeddings
(*UnSimCSE*)

- Supervised Simple Contrastive Learning
of Sentence Embeddings (*SuSimCSE*)

- Cased BERT Model for Scientific Text
(*SciBERT-cased*)

- Uncased BERT Model for Scientific Text
(*SciBERT-uncased*)

- BERT Model with Natural Language
Interface (*BERT-NLI*)

- BERT Model for Paraphrase (*BERT-PP*)

- Scientific Paper Embeddings using Citation-informed Transformers
(*SPECTER*)
### requirements.txt ###
---------------
This file contains python packages which we have employed for building our IR model.

### QBE-KLDivLoss.ipynb ###
---------------
The file QBE-KLDivLoss.ipynb contains code for our IR model trained using KLDivLoss.

More on KLDivLoss can be found here: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
### QBE-NLLLoss.ipynb ###
---------------

The file QBE-KLDivLoss.ipynb contains code for our IR model trained using NLLLoss.

More on NLLLoss can be found here: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html

## Results ##

The results can be found in the report.
