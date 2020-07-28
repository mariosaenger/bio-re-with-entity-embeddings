import os
import pandas as pd

from sklearn.pipeline import Pipeline

from data.disease_ontology import DiseaseOntology
from data.pubtator import PubtatorPreparationUtils, Pubtator
from data.resources import PUBTATOR_MUT2PUB_FILE, MESH_TO_DRUGBANK_MAPPING, PUBTATOR_CHEM2PUB_FILE, DO_ONTOLOGY_FILE, \
    PUBTATOR_DIS2PUB_FILE
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu, MeshTermToDoidMapper, MeshTermToDrugBankIdMapper


class EntityDataSetPreparation(LoggingMixin):

    def __init__(self):
        super(EntityDataSetPreparation, self).__init__()

    def run(self, working_dir: str):
        raise NotImplementedError("run() has to be implemented by sub-classes")


class PubtatorMutationDataSetPreparation(EntityDataSetPreparation):

    def __init__(self):
        super(PubtatorMutationDataSetPreparation, self).__init__()

    def run(self, working_directory: str):
        self.log_info("Start preparation of PubMed mutation occurrences")

        entity_ds_dir = os.path.join(working_directory, "mutation")
        os.makedirs(entity_ds_dir, exist_ok=True)

        pubtator = Pubtator()

        self.log_info("Read mutation data from %s", PUBTATOR_MUT2PUB_FILE)
        mutation_overview = pubtator.build_mutation_overview(PUBTATOR_MUT2PUB_FILE)
        self.log_info("Found %s mutations in total", len(mutation_overview))

        mutation_instances_file = os.path.join(entity_ds_dir, "instances.tsv")
        mutation_pubmed_ids_file = os.path.join(entity_ds_dir, "pubmed_ids.txt")

        self.log_info("Saving mutation information to entity data set directory")
        mutation_pipeline = Pipeline([
            ("ConvertArticlesToString", pdu.map("articles", PubtatorPreparationUtils.set_to_string, "articles_str")),
            ("ExtractPubMedIds", pdu.extract_unique_values("articles", mutation_pubmed_ids_file)),
            ("RenameIdColumn", pdu.rename_columns({"rs_identifier": "entity_id"})),
            ("SaveInstancesAsTsv", pdu.to_csv(mutation_instances_file, columns=["entity_id", "articles_str"]))
        ])

        mutation_overview = mutation_pipeline.fit_transform(mutation_overview)
        mutation_overview.to_csv(mutation_instances_file, sep="\t", columns=["entity_id", "articles_str"], index=False)

        self.log_info("Create mapping from PubMed id to mutation id")
        pubmed_to_mutation = PubtatorPreparationUtils.create_pubmed_id_to_entity_map(mutation_overview)
        pipeline = Pipeline([
            ("ConvertEntityIdsToString", pdu.map("entity_ids", PubtatorPreparationUtils.set_to_string, "entity_ids_str")),
        ])
        pubmed_to_mutation = pipeline.fit_transform(pubmed_to_mutation)

        pubmed2entity_file = os.path.join(entity_ds_dir, "pubmed2entity.tsv")
        pubmed_to_mutation.to_csv(pubmed2entity_file,  sep="\t", columns=["entity_ids_str"], index_label="pubmed_id")


class PubtatorDrugOccurrencesPreparation(EntityDataSetPreparation):

    def __init__(self):
        super(PubtatorDrugOccurrencesPreparation, self).__init__()

    def run(self, working_directory: str):
        self.log_info("Start preparation of drug occurrences")

        entity_ds_dir = os.path.join(working_directory, "drug")
        os.makedirs(entity_ds_dir, exist_ok=True)

        pubtator = Pubtator()

        mesh_to_drugbank_mapping = pd.read_csv(MESH_TO_DRUGBANK_MAPPING, sep="\t", )

        # Drug data
        self.log_info("Read disease data from %s", PUBTATOR_CHEM2PUB_FILE)
        drug_data = pubtator.build_drug_overview(PUBTATOR_CHEM2PUB_FILE)
        self.log_info("Found %s unique diseases in total", len(drug_data))

        drug_instances_file = os.path.join(entity_ds_dir, "instances.tsv")
        drug_pubmed_ids_file = os.path.join(entity_ds_dir, "pubmed_ids.txt")

        self.log_info("Saving disease information to entity data set directory")

        pipeline = Pipeline([
            ("MapMashTermToDrugBankId", MeshTermToDrugBankIdMapper(mesh_to_drugbank_mapping)),
            ("ConvertArticlesToString", pdu.map("articles", PubtatorPreparationUtils.set_to_string, "articles_str")),
            ("RenameIdColumn", pdu.rename_columns({"MeshID": "entity_id"})),
            ("ExtractPubMedIds", pdu.extract_unique_values("articles", drug_pubmed_ids_file)),
            ("SaveInstancesAsTsv", pdu.to_csv(drug_instances_file, columns=["entity_id", "articles_str"]))
        ])

        drug_data = pipeline.fit_transform(drug_data)

        instances_file = os.path.join(entity_ds_dir, "instances.tsv")
        drug_data.to_csv(instances_file, sep="\t", columns=["entity_id", "articles_str"], index=False)

        self.log_info("Create mapping from PubMed id to Drug id")
        pubmed_to_drug = PubtatorPreparationUtils.create_pubmed_id_to_entity_map(drug_data)
        pipeline = Pipeline([
            ("ConvertEntityIdsToString",
             pdu.map("entity_ids", PubtatorPreparationUtils.set_to_string, "entity_ids_str")),
        ])
        pubmed_to_drug = pipeline.fit_transform(pubmed_to_drug)

        pubmed2entity_file = os.path.join(entity_ds_dir, "pubmed2entity.tsv")
        pubmed_to_drug.to_csv(pubmed2entity_file, sep="\t", columns=["entity_ids_str"], index_label="pubmed_id")


class PubtatorDiseaseOccurrencesPreparation(EntityDataSetPreparation):

    def __init__(self):
        super(PubtatorDiseaseOccurrencesPreparation, self).__init__()

    def run(self, working_directory: str):
        self.log_info("Start preparation of disease occurrences")

        entity_ds_dir = os.path.join(working_directory, "disease")
        os.makedirs(entity_ds_dir, exist_ok=True)

        self.disease_ontology = DiseaseOntology(DO_ONTOLOGY_FILE)
        pubtator = Pubtator()

        # Disease data
        self.log_info("Read disease data from %s", PUBTATOR_DIS2PUB_FILE)
        disease_overview = pubtator.build_disease_overview(PUBTATOR_DIS2PUB_FILE)
        self.log_info("Found %s unique diseases in total", len(disease_overview))

        disease_instances_file = os.path.join(entity_ds_dir, "instances.tsv")
        disease_pubmed_ids_file = os.path.join(entity_ds_dir, "pubmed_ids.txt")

        self.log_info("Saving disease information to entity data set directory")
        disease_pipeline = Pipeline([
            ("MapMashTermTodoDoid", MeshTermToDoidMapper(self.disease_ontology)),
            ("FilterEntriesWithoutDoid", pdu.not_null("doid")),

            ("ConvertArticlesToString", pdu.map("articles", PubtatorPreparationUtils.set_to_string, "articles_str")),
            ("RenameIdColumn", pdu.rename_columns({"doid": "entity_id"})),

            ("ExtractPubMedIds", pdu.extract_unique_values("articles", disease_pubmed_ids_file)),
            ("SaveInstancesAsTsv", pdu.to_csv(disease_instances_file, columns=["entity_id", "articles_str"]))
        ])

        disease_pipeline.fit_transform(disease_overview)

        self.log_info("Create mapping from PubMed id to disease id")
        pubmed_to_disease = PubtatorPreparationUtils.create_pubmed_id_to_entity_map(disease_overview)
        pipeline = Pipeline([
            ("ConvertEntityIdsToString", pdu.map("entity_ids", PubtatorPreparationUtils.set_to_string, "entity_ids_str")),
        ])
        pubmed_to_disease = pipeline.fit_transform(pubmed_to_disease)

        pubmed2entity_file = os.path.join(entity_ds_dir, "pubmed2entity.tsv")
        pubmed_to_disease.to_csv(pubmed2entity_file,  sep="\t", columns=["entity_ids_str"], index_label="pubmed_id")
