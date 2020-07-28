# Large-scale entity and entity pair embeddings
This repository contains source code to learn dense semantic representations for biomedical 
entities and pairs of entities as used in Sänger and Leser: "Large-scale Entity Representation 
Learning for Biomedical Relationship Extraction" (Bioinformatics, 2020). 

The approach aims to perform biomedical relation extraction on corpus-level based on entity and 
entity pair embeddings learned on the complete PubMed corpus. For this we use focus on all articles 
mentioning a certain biomedical entity (e.g. mutation <i>V600E</i>) or pair of entities within the article 
title or abstract. We concatenate all articles mention the entity / entity pair and apply <i>Doc2Vec</i> 
to learn an embedding for each distinct entity resp. pair of entities.

## Usage
For the computing entity and entity pair embeddings we utilize the complete PubMed corpus and make 
use of the data and entity annotations provided by 
<a href="https://www.ncbi.nlm.nih.gov/research/pubtator/" target="_blank">Pubtator</a>.

#### Learn entity embeddings
Learning entity embeddings can be done in two steps:
* Prepare the entity annotations:
~~~
python prepare_entity_dataset.py --working_dir <dir> --entity_type mutation
~~~
We support entity types <i>disease</i>, <i>drug</i>, and <i>mutation</i>.

* Run representation learning:
~~~
python learn_doc2vec.py
~~~
Example configurations can be found in <i>resources/configurations</i>.

#### Learn entity pair embeddings
To learn entity pair embeddings, preparation of the entity annotations has to be performed 
first (see above).


## Citation
Please use the following bibtex entry to cite our work:
```
@article{saenger2020entityrepresentation,
  title={Large-scale Entity Representation Learning for Biomedical Relationship Extraction},
  author={S{\"a}nger, Mario and Leser, Ulf},
  journal={Bioinformatics},
  year={2020},
  publisher={Oxford University Press}
}
```

## Acknowledgements


