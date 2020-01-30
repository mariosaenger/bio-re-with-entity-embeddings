# Large-scale biomedical relation extraction with entity and pair embeddings
This repository contains the source code for the large-scale biomedical relation 
extraction with entity and pair embeddings project. The project aims to perform 
biomedical relation extraction on corpus-level based on entity and entity pair embeddings
learned with Doc2Vec.

## Goldstandard preparation
In the project we used [CIViC](https://civicdb.org/home), [DoCM](http://www.docm.info/) 
and [PharmaGKB](https://www.pharmgkb.org/) as gold standard. Necessary preparation steps
to use the data sets in this project can be performed by running

    python -m data.civic
    python -m data.docm
    python -m data.pharma

The location of the data source resources can be configured in <code>data/resources.py</code>.

## Basic preparation
Training of entity and entity pair embeddings is based on annotations from 
[Pubtator](ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/). Download and extract 
<code>bioconcepts2pubtator_offsets.gz</code>, <code>disease2pubtator.gz</code> and 
<code>mutation2pubtator.gz</code> and specify the location of the resources
in <code>data/resources.py</code>.

To prepare the resources for the training process the following steps have to
be run:
- Basic pubtator preparation

   <code>python prepare_pubtator.py <output-dir></code>

- Prepare and integrate gold standard data sets

    <code>python prepare_mutation_disease.py <output-dir></code>

## Learning of entity embeddings
For computing entity embeddings for diseases and mutations the following steps have 
to be performed:
   
- Extract mutation and disease articles

~~~ 
   python extract_articles.py <output-dir>/entity-ds/disease_pubmed_ids.txt  \
                               <output-dir>/entity-ds/disease_articles.txt \
                              16 2000
 
   python extract_articles.py <output-dir>/entity-ds/mutation_pubmed_ids.txt  \
                               <output-dir>/entity-ds/mutation_articles.txt \
                              16 2000 
~~~

- Prepare the articles for Doc2Vec learning
~~~
    python prepare_doc2vec_input.py <output-dir>/entity-ds/disease_instances.tsv \
                                      doid articles_str \
                                      <output-dir>/entity-ds/disease_articles.txt \
                                      <output-dir>/entity-ds/disease_doc2vec.txt

     python prepare_doc2vec_input.py <output-dir>/entity-ds/mutation_instances.tsv \
                                      rs_identifier articles_str \
                                      <output-dir>/entity-ds/mutation_articles.txt \
                                      <output-dir>/entity-ds/mutation_doc2vec.txt
~~~

- Run embedding training (for example 500-dimensional embeddings)

~~~
    python learn_doc2vec.py <output-dir>/entity-ds/disease_doc2vec.txt \
                             ../configurations/doc2vec-0500.config \
                             <output-dir>/embeddings/ disease-embeddings-v0500

     python learn_doc2vec.py <output-dir>/entity-ds/mutation_doc2vec.txt \
                               ../configurations/doc2vec-0500.config \
                               <output-dir>/embeddings/ mutation-embeddings-v0500
~~~


## Learning of entity pair embeddings
For computing entity pair embeddings for diseases and mutations the following steps have 
to be performed:

- Extract mutation and disease articles

~~~
    python extract_articles.py <output-dir>>/pair-ds/pair_pubmed_ids.txt  \
                                <output-dir>/pair-ds/pair_articles.txt \
                                16 2000
~~~

- Prepare the articles for Doc2Vec learning

~~~
    python prepare_doc2vec_input.py <outputdir>/pair-ds/pair_instances.tsv \
                                     "source_id target_id" articles_str 
                                     <output-dir>/pair-ds/pair_articles.tsv \
                                     <output-dir>/pair-ds/pair_doc2vec.txt
~~~

- Run embedding training (for example 500-dimensional embeddings)
~~~
    python learn_doc2vec.py <output-dir>/pair-ds/pair_doc2vec.txt \
                            ../configurations/doc2vec-0500.config  \
                            <output-dir>/embeddings/ mutation-embeddings-v0500
~~~


## Run experiments
To run the classification experiments the following options exist:
~~~
    python run_experiments.py md <output-dir> _embs entity _result
~~~
