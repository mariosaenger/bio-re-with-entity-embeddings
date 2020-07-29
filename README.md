# Large-scale entity and entity pair embeddings
This repository contains source code to learn dense semantic representations for biomedical 
entities and pairs of entities as used in Sänger and Leser: "Large-scale Entity Representation 
Learning for Biomedical Relationship Extraction" (Bioinformatics, 2020). 

The approach aims to perform biomedical relation extraction on corpus-level based on entity and 
entity pair embeddings learned on the complete PubMed corpus. For this we use focus on all articles 
mentioning a certain biomedical entity (e.g. mutation <i>V600E</i>) or pair of entities within the article 
title or abstract. We concatenate all articles mention the entity / entity pair and apply paragraph vectors
(<i>Le and Mikolov, 2014</i>) to learn an embedding for each distinct entity resp. pair of entities.

| [Usage](#usage) | [Supported Entity Types](#supported-entity-types) | [Citation](#citation) | [Acknowledgements](#acknowledgements) |
 
## Usage
For the computing entity and entity pair embeddings we utilize the complete PubMed corpus and make 
use of the data and entity annotations provided by 
<a href="https://www.ncbi.nlm.nih.gov/research/pubtator/" target="_blank">PubTator Central</a>.

#### Download resources
* Download annotations from PubTator Central:
~~~
python download_resources.py --resources pubtator_central
~~~
<i>Note: The annotation data requires > 70GB of disk space.</i> 

#### Learn entity embeddings
Learning entity embeddings can be done in two steps:
* Prepare entity annotations:
~~~
python prepare_entity_dataset.py --working_dir _out --entity_type mutation
~~~
We support entity types <i>disease</i>, <i>drug</i>, and <i>mutation</i>.

* Run representation learning:
~~~
python learn_embeddings.py --input_file _out/mutation/doc2vec_input.txt \
                           --config_file ../resources/configurations/doc2vec-0500.config \
                           --model_name mutation-v0500 \
                           --output_dir _out/mutation  
~~~
Example configurations can be found in <i>resources/configurations</i>.

#### Learn entity pair embeddings
To learn entity pair embeddings, preparation of the entity annotations has to be performed 
first (see above). Analogously to the entity embeddings, learning of pair embeddings is 
performed in two steps:
* Prepare pair annotations:
~~~
python prepare_pair_dataset.py --working_dir _out --source_type mutation --target_type disease
~~~
We support entity types <i>disease</i>, <i>drug</i>, and <i>mutation</i>.

* Run representation learning:
~~~
python learn_embeddings.py --input_file _out/mutation-disease/doc2vec_input.txt \
                           --config_file ../resources/configurations/doc2vec-0500.config \
                           --model_name mutation-disease-v0500 \
                           --output_dir _out/mutation-disease  
~~~
Example configurations can be found in <i>resources/configurations</i>.

## Supported entity types

| Entity Type  | Identifier  | Example  |
|---|---|---|
| Disease  | MeSH  | MESH:D006984 (<i>hypertrophic chondrocytes</i>) |
|   |  Disease Ontology ID (DOID) <sup id="a1">[1](#f1)</sup> | DOID:60155 (<i>visual agnosia</i>)  |
| Drug  | Drugbank ID  | DB00166 (<i>lipoic acid</i>)  |
| Mutation  | RS-Identifier  | rs1356828811 (<i>E64D</i>)  |

<a id="f1">1</a>: Use option "<i>--entity_type disease-doid</i>" when calling `prepare_entity_dataset.py` to normalize 
disease annotations to the Disease Ontology.  



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
* We use the annotations from <a href="https://www.ncbi.nlm.nih.gov/research/pubtator/" target="_blank">PubTator Central</a> 
to compute the entity embeddings. For further details see [here](https://pubmed.ncbi.nlm.nih.gov/31114887/) and refer to:

  Wei, Chih-Hsuan, et al. "<i>PubTator central: automated concept annotation for biomedical full text articles.</i>" 
  Nucleic acids research 47.W1 (2019): W587-W593.
 
* We use information from the <a href="https://disease-ontology.org/">Disease Ontology</a> to normalize disease annotations. For 
further details see [here](https://pubmed.ncbi.nlm.nih.gov/30407550/) and refer to:

  Schriml, Lynn M., et al. "<i>Human Disease Ontology 2018 update: classification, content and workflow expansion.</i>" 
  Nucleic acids research 47.D1 (2019): D955-D962. 

* We use the paragraph vectors model to perform entity representation learning. 
For further details see [here](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) and refer to:
  
  Le, Quoc, and Tomas Mikolov. "<i>Distributed representations of sentences and documents.</i>" 
  International conference on machine learning. 2014.
 

