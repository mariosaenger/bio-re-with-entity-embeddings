import os

# Specify location of pubtator files
PUBTATOR_FOLDER = "../../pubtator_20190105"
PUBTATOR_MUT2PUB_FILE = os.path.join(PUBTATOR_FOLDER, "mutation2pubtator")
PUBTATOR_DIS2PUB_FILE = os.path.join(PUBTATOR_FOLDER, "disease2pubtator")
PUBTATOR_OFFSET_FILE = os.path.join(PUBTATOR_FOLDER, "bioconcepts2pubtator_offsets")

# Specify location of CIViC files
CIVIC_DIR = "_resources/civic"
CIVIC_EVIDENCES_FILE = os.path.join(CIVIC_DIR, "ClinicalEvidenceSummaries.tsv")
CIVIC_VARIANT_SUMMARY_FILE = os.path.join(CIVIC_DIR, "VariantSummaries.tsv")
CIVIC_DATA_SET_FILE = os.path.join(CIVIC_DIR, "civic.tsv")

# Specify location of DoCM files
DOCM_DIR = "_resources/docm/"
DOCM_VARIANTS = os.path.join(DOCM_DIR, "variants.tsv")
DOCM_HGVS_MAPPING = os.path.join(DOCM_DIR, "hgvs_mapping.txt")
DOCM_DATA_SET_FILE = os.path.join(DOCM_DIR, "docm.tsv")

# Specify location of PharmaGKB files
PHARMA_DIR = "_resources/pharma"
PHARMA_RELATIONSHIP_FILE = os.path.join(PHARMA_DIR, "relationships.tsv")
PHARMA_DATA_SET_FILE = os.path.join(PHARMA_DIR, "pharma.tsv")

# Specify location of drugbank file
DRUGBANK_DIR = "_resources/drugbank"
DRUGBANK_FILE = os.path.join(DRUGBANK_DIR, "full_database.xml")

# Specify disease ontology files
DO_DIR = "_resources/do"
DO_DOID_FILE = os.path.join(DO_DIR, "doid.obo")
DO_ONTOLOGY_FILE = os.path.join(DO_DIR, "ontology.tsv")
DO_CANCER_ONTOLOGY_FILE = os.path.join(DO_DIR, "ontology_cancer.tsv")

# General resources
RSID_TO_HGVS_FILE = "_resources/rsid_to_hgvs.txt"

