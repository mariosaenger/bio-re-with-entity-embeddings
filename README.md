# Large-scale entity and entity pair embeddings
This repository contains source code to learn dense semantic representations for biomedical 
entities and pairs of entities as used in [Sänger and Leser: "Large-scale Entity Representation 
Learning for Biomedical Relationship Extraction" (Bioinformatics, 2020)](https://doi.org/10.1093/bioinformatics/btaa674). 

The approach aims to perform biomedical relation extraction on corpus-level based on entity and 
entity pair embeddings learned on the complete PubMed corpus. For this we use focus on all articles 
mentioning a certain biomedical entity (e.g. mutation <i>V600E</i>) or pair of entities within the article 
title or abstract. We concatenate all articles mention the entity / entity pair and apply paragraph vectors
(<i>Le and Mikolov, 2014</i>) to learn an embedding for each distinct entity resp. pair of entities.

__Content:__ [Usage](#usage) | [Pre-trained Entity Embeddings](#pre-trained-entity-embeddings) | [Embedding Training](#train-your-own-embeddings) | [Supported Entity Types](#supported-entity-types) | [Citation](#citation) | [Acknowledgements](#acknowledgements) |

## Usage
The implementation of the embeddings is based on [Gensim](https://radimrehurek.com/gensim/). The following snippet highlights the basic use
of the pre-trained embeddings.   
```python
from gensim.models import KeyedVectors

# Loading pre-trained entity model
model = KeyedVectors.load("mutation-v0500.bin")

# Print number of distinct entities of the model
print(f"Distinct entities: {len(model.vocab)}\n")

# Get the embedding for an specific entity
entity_embedding = model["rs113488022"]
print(f"Embedding of rs113488022:\n{entity_embedding}\n")

# Find similar entities
print("Most similar entities to rs113488022:")
top5_nearest_neighbors = model.most_similar("rs113488022", topn=5)
for i, (entity_id, sim) in enumerate(top5_nearest_neighbors):
    print(f" {i+1}: {entity_id} (similarity: {sim:.3f})")
```
This should output:
```
Distinct entities: 47498

Embedding of rs113488022:
[ 1.15715809e-01  4.90018785e-01 -6.05004542e-02 -8.35603476e-02
  9.20398310e-02 -1.51171118e-01  4.01901715e-02 -2.36775234e-01
  ...
]

Most similar entities to rs113488022:
 1: rs121913227 (similarity: 0.690)
 2: rs121913364 (similarity: 0.628)
 3: rs121913529 (similarity: 0.610)
 4: rs121913357 (similarity: 0.573)
 5: rs11554290 (similarity: 0.571)
```

## Pre-trained Entity Embeddings
<table>
    <tr>
        <th>Entity Type</th>
        <th>Identifier</th>
        <th style="text-align: right">#Entities</th>
        <th style="text-align: center">Vocabulary</th>
        <th>v500</th>
        <th>v1000</th>
        <th>v1500</th>
        <th>v2000</th>
    </tr>
    <tr>
        <td>Cellline</td>
        <td>Cellosaurus ID</td>
        <td style="text-align: right">4,654</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/cellline/cellline-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/cellline/cellline-v0500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/cellline/cellline-v1000.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/cellline/cellline-v1500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/cellline/cellline-v2000.bin" download target="_blank">Vectors</a></td>
    </tr>
    <tr>
        <td>Chemical</td>
        <td>MeSH</td>
        <td style="text-align: right">109,716</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/chemical/chemical-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/chemical/chemical-v0500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/chemical/chemical-v1000.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/chemical/chemical-v1500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/chemical/chemical-v2000.bin" download target="_blank">Vectors</a></td>
    </tr>
    <tr>
        <td>Disease</td>
        <td>MeSH</td>
        <td style="text-align: right">10,712</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease/disease-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease/disease-v0500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease/disease-v1000.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease/disease-v1500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease/disease-v2000.bin" download target="_blank">Vectors</a></td>
    </tr>
        <tr>
        <td></td>
        <td>DOID</td>
        <td style="text-align: right">3,157</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease-doid/disease-doid-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease-doid/disease-doid-v0500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease-doid/disease-doid-v1000.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease-doid/disease-doid-v1500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/disease-doid/disease-doid-v2000.bin" download target="_blank">Vectors</a></td>
    </tr>
    <tr>
        <td>Drug</td>
        <td>Drugbank ID</td>
        <td style="text-align: right">5,966</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/drug/drug-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/drug/drug-v0500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/drug/drug-v1000.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/drug/drug-v1500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/drug/drug-v2000.bin" download target="_blank">Vectors</a></td>
    </tr>
    <tr>
        <td>Gene</td>
        <td>NCBI Gene ID</td>
        <td style="text-align: right">171,686</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/gene/gene-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/gene/gene-v0500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/gene/gene-v1000.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/gene/gene-v1500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/gene/gene-v2000.bin" download target="_blank">Vectors</a></td>
    </tr>
    <tr>
        <td>Mutation</td>
        <td>RS-Identifier</td>
        <td style="text-align: right">47,498</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/mutation/mutation-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/mutation/mutation-v0500.bin dowload">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/mutation/mutation-v1000.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/mutation/mutation-v1500.bin" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/mutation/mutation-v2000.bin" download target="_blank">Vectors</a></td>
    </tr>
    <tr>
        <td>Species</td>
        <td>NCBI Taxonomy</td>
        <td style="text-align: right">176,989</td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/species/species-v0500.vocab">Vocab</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/species/species-v0500.bin" download target="_blank" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/species/species-v1000.bin" download target="_blank" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/species/species-v1500.bin" download target="_blank" download target="_blank">Vectors</a></td>
        <td style="text-align: center"><a href="https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/species/species-v2000.bin" download target="_blank" download target="_blank">Vectors</a></td>
    </tr>
</table>

## Train your own embeddings
For the computing entity and entity pair embeddings we utilize the complete PubMed corpus and make 
use of the data and entity annotations provided by [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator/).

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
We support entity types <i>cell line</i>, <i>chemical</i>, <i>disease</i>, <i>drug</i>, 
<i>gene</i>, <i>mutation</i>, and <i>species</i>.

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
| Cell line  | Cellosaurus ID  | CVCL:0027 (<i>Hep-G2</i>)  |
| Chemical  | MeSH  | MESH:D000068878 (<i>hTrastuzumab</i>) |
| Disease  | MeSH  | MESH:D006984 (<i>hypertrophic chondrocytes</i>) |
|          |  Disease Ontology ID (DOID) <sup id="a1">[1](#f1)</sup> | DOID:60155 (<i>visual agnosia</i>)  |
| Drug  | Drugbank ID  | DB00166 (<i>lipoic acid</i>)  |
| Gene  | NCBI Gene ID  | NCBI:673 (<i>BRAF</i>)  |
| Mutation  | RS-Identifier  | rs113488022 (<i>V600E</i>)  | 
| Species  | NCBI Taxonomy  | TAXON:9606 (<i>human</i>)  | 

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
 

