## Drug-Drug
Here you can find a description, how to learn entity and entity pair embeddings for the drug-drug scenario.

### Basic preparation
Training of entity and entity pair embeddings is based on annotations from 
[GeneView](http://bc3.informatik.hu-berlin.de/). 

To prepare the resources for the training process the following steps have to
be run:

    python prepare_drug_drug.py <output-dir>

### Learning of entity embeddings
For computing drug entity embeddings the following steps have  to be performed:
   
- Extract drug articles
~~~ 
   python extract_articles.py <output-dir>/geneview/drug_pubmed_ids.txt  \
                              <output-dir>/geneview/drug_articles.txt \
                              16 2000
~~~

- Prepare the articles for Doc2Vec learning
~~~
    python prepare_doc2vec_input.py <output-dir>/geneview/geneview.csv \
                                     drug_id articles_str \
                                     <output-dir>/geneview/drug_articles.txt \
                                     <output-dir>/geneview/drug_doc2vec.txt
~~~

- Run embedding training (for example 500-dimensional embeddings)

~~~
    python learn_doc2vec.py <output-dir>/geneview/drug_doc2vec.txt \
                              ../configurations/doc2vec-0500.config \
                              <output-dir>/embeddings/ drug-v0500
~~~


### Learning of entity pair embeddings
For computing drug-drug pair embeddings the following steps have to be performed:

- Extract drug-drug PubMed articles
~~~
    python extract_articles.py <output-dir>>/geneview/drug_pubmed_ids.txt  \
                                <output-dir>/geneview/drug_articles.txt \
                                16 2000
~~~

- Prepare the articles for Doc2Vec learning

~~~
    python prepare_doc2vec_input.py <outputdir>/geneview/pair_instances.tsv \
                                     "source_id target_id" articles_str
                                     <output-dir>/geneview/drug_articles.txt \
                                     <output-dir>/geneview/pair_doc2vec.txt
~~~

- Run embedding training (for example 500-dimensional embeddings)
~~~
    python learn_doc2vec.py <output-dir>/geneview/pair_doc2vec.txt \
                            ../configurations/doc2vec-0500.config  \
                            <output-dir>/embeddings/ drug-drug-v0500
~~~


### Run experiments
To run the classification experiments the following options exist:
- Run entity embeddings experiments
~~~
    python run_experiments.py md <output-dir> <embedding-folder> entity _result
~~~
- Run pair embeddings experiments
~~~
    python run_experiments.py md <output-dir> <embedding-folder> pair _result
~~~

