## Mutation-Disease
Here you can find a description, how to learn entity and entity pair embeddings for the mutation-disease scenario.

### Basic preparation
Training of entity and entity pair embeddings is based on annotations from 
[Pubtator](ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/). Download and extract 
<code>bioconcepts2pubtator_offsets.gz</code>, <code>disease2pubtator.gz</code> and 
<code>mutation2pubtator.gz</code> and specify the location of the resources
in <code>data/resources.py</code>.

To prepare the resources for the training process the following steps have to
be run:
- Basic pubtator preparation

   <code>python prepare_pubtator.py <output-dir></code>

### Learning of entity embeddings
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
                             <output-dir>/embeddings/ disease-v0500

     python learn_doc2vec.py <output-dir>/entity-ds/mutation_doc2vec.txt \
                               ../configurations/doc2vec-0500.config \
                               <output-dir>/embeddings/ mutation-v0500
~~~


### Learning of entity pair embeddings
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
                            <output-dir>/embeddings/ mutation-disease-v0500
~~~
