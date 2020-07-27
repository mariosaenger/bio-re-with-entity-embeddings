import os

# Specify location of pubtator files
PUBTATOR_FOLDER = "../../pubtator_20190105"
PUBTATOR_MUT2PUB_FILE = os.path.join(PUBTATOR_FOLDER, "mutation2pubtator")
PUBTATOR_DIS2PUB_FILE = os.path.join(PUBTATOR_FOLDER, "disease2pubtator")
PUBTATOR_CHEM2PUB_FILE = os.path.join(PUBTATOR_FOLDER, "chemical2pubtator")
PUBTATOR_OFFSET_FILE = os.path.join(PUBTATOR_FOLDER, "bioconcepts2pubtator_offsets")

# Specify disease ontology files
DO_DIR = "_resources/do"
DO_DOID_FILE = os.path.join(DO_DIR, "doid.obo")
DO_ONTOLOGY_FILE = os.path.join(DO_DIR, "ontology.tsv")
DO_CANCER_ONTOLOGY_FILE = os.path.join(DO_DIR, "ontology_cancer.tsv")

# Specify CTD files
MESH_TO_DRUGBANK_MAPPING = "../resources/mappings/drug_mapping.tsv"
