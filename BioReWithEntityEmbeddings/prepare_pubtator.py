import argparse
import os
import pandas as pd

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from data.disease_ontology import DiseaseOntology
from data.pubtator import Pubtator
from data.resources import DO_ONTOLOGY_FILE, DO_CANCER_ONTOLOGY_FILE, PUBTATOR_DIS2PUB_FILE, PUBTATOR_MUT2PUB_FILE
from prepare_mutation_disease import MeshTermToDoidMapper, CancerDiseaseFilter
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu


class PubtatorPreparation(LoggingMixin):

    def __init__(self):
        super(PubtatorPreparation, self).__init__()

    def run(self, working_directory: str):
        self.log_info("Start preparation of pubtator information")

        # Create sub folders for the results
        pubtator_dir = os.path.join(working_directory, "pubtator")
        os.makedirs(pubtator_dir, exist_ok=True)

        entity_ds_dir = os.path.join(working_directory, "entity-ds")
        os.makedirs(entity_ds_dir, exist_ok=True)

        pair_ds_dir = os.path.join(working_directory, "pair-ds")
        os.makedirs(pair_ds_dir, exist_ok=True)

        co_occurrences_file = os.path.join(pubtator_dir, "pubtator_co_occurrences.csv")

        self.disease_ontology = DiseaseOntology(DO_ONTOLOGY_FILE)
        self.cancer_ontology = DiseaseOntology(DO_CANCER_ONTOLOGY_FILE)

        pubtator = Pubtator()

        # Disease data
        self.log_info("Read disease data from %s", PUBTATOR_DIS2PUB_FILE)
        disease_overview = pubtator.build_disease_overview(PUBTATOR_DIS2PUB_FILE)
        disease_overview.to_csv(os.path.join(pubtator_dir, "pubtator_diseases.csv"), sep="\t", index_label="disease_id")
        self.log_info("Found %s unique diseases in total", len(disease_overview))

        self.log_info("Create mapping from pubmed id to disease id")
        pubmed_to_disease = self.create_pubmed_id_to_entity_map(disease_overview)
        pubmed_to_disease.to_csv(os.path.join(pubtator_dir, "pubtator_pubmed2diseases.csv"), sep="\t", index_label="pubmed_id")

        disease_instances_file = os.path.join(entity_ds_dir, "disease_instances.tsv")
        disease_pubmed_ids_file = os.path.join(entity_ds_dir, "disease_pubmed_ids.txt")

        self.log_info("Saving disease information to entity data set directory")
        mesh_extractor = lambda id, row: id
        id_extractor = lambda id, row: id
        disease_pipeline = Pipeline([
            ("MapMashTermTodoDoid", MeshTermToDoidMapper(self.disease_ontology, mesh_extractor, id_extractor)),
            ("FilterEntriesWithoutDoid", pdu.not_null("doid")),

            ("ConvertArticlesToString", pdu.map("articles", self.set_to_string, "articles_str")),

            ("ExtractPubMedIds", pdu.extract_unique_values("articles", disease_pubmed_ids_file)),
            ("SaveInstancesAsTsv", pdu.to_csv(disease_instances_file, columns=["doid", "articles_str"]))
        ])
        diseases_df = disease_pipeline.fit_transform(disease_overview)

        cancer_instances_file = os.path.join(entity_ds_dir, "cancer_instances.tsv")
        cancer_pubmed_ids_file = os.path.join(entity_ds_dir, "cancer_pubmed_ids.txt")

        self.log_info("Saving cancer information to entity data set directory")
        cancer_pipeline = Pipeline([
            ("FilterCancerDisease", CancerDiseaseFilter(self.cancer_ontology.get_all_doids())),
            ("ExtractPubMedIds", pdu.extract_unique_values("articles", cancer_pubmed_ids_file)),
            ("SaveInstancesAsTsv", pdu.to_csv(cancer_instances_file, columns=["doid", "articles_str"]))
        ])
        cancer_pipeline.fit_transform(diseases_df)

        # Mutation data
        self.log_info("Read mutation data from %s", PUBTATOR_MUT2PUB_FILE)
        mutation_overview = pubtator.build_mutation_overview(PUBTATOR_MUT2PUB_FILE)
        mutation_overview.to_csv(os.path.join(pubtator_dir, "pubtator_mutations.csv"), sep="\t", index_label="mutation_id")
        self.log_info("Found %s mutations in total", len(mutation_overview))

        self.log_info("Create mapping from pubmed id to mutation id")
        pubmed_to_mutation = self.create_pubmed_id_to_entity_map(mutation_overview)
        pubmed_to_mutation.to_csv(os.path.join(pubtator_dir, "pubtator_pubmed2mutations.csv"), sep="\t", index_label="pubmed_id")

        mutation_instances_file = os.path.join(entity_ds_dir, "mutation_instances.tsv")
        mutation_pubmed_ids_file = os.path.join(entity_ds_dir, "mutation_pubmed_ids.txt")

        self.log_info("Saving mutation information to entity data set directory")
        mutation_pipeline = Pipeline([
            ("ConvertArticlesToString", pdu.map("articles", self.set_to_string, "articles_str")),
            ("ExtractPubMedIds", pdu.extract_unique_values("articles", mutation_pubmed_ids_file)),
            ("SaveInstancesAsTsv", pdu.to_csv(mutation_instances_file, columns=["rs_identifier", "articles_str"]))
        ])
        mutation_pipeline.fit_transform(mutation_overview)

        self.log_info("Build co-occurrence mapping")
        pub_co_occurrences = self.find_entity_cooccurrences(pubmed_to_mutation, pubmed_to_disease)

        pair_instances_file = os.path.join(pair_ds_dir, "pair_instances.tsv")
        pair_pubmed_ids_file = os.path.join(pair_ds_dir, "pair_pubmed_ids.txt")

        cancer_pair_instances_file = os.path.join(pair_ds_dir, "cancer_pair_instances.tsv")
        cancer_pair_pubmed_ids_file = os.path.join(pair_ds_dir, "cancer_pair_pubmed_ids.txt")

        mesh_extractor = lambda id, row: id.split("#")[1]
        id_extractor = lambda id, row: row["id1"]

        co_occurrence_pipeline = Pipeline([
            ("MapMashTermToDoid", MeshTermToDoidMapper(self.disease_ontology, mesh_extractor, id_extractor)),
            ("FilterEntriesWithDoid", pdu.not_null("doid")),
            ("ConvertArticlesToString", pdu.map("articles", self.set_to_string, "articles_str")),

            ("DropDuplicates1", pdu.drop_duplicates(["id1", "doid"])),
            ("DropDuplicates2", pdu.drop_duplicates(["id1", "doid", "articles_str"])),

            ("RenameColumnNames", pdu.rename_columns({"id1": "source_id", "doid": "target_id"})),

            ("ExtractPubMedIds", pdu.extract_unique_values("articles", pair_pubmed_ids_file)),
            ("SaveInstancesAsTsv",
             pdu.to_csv(pair_instances_file, columns=["source_id", "target_id", "articles_str"])),
        ])

        pub_co_occurrences = co_occurrence_pipeline.fit_transform(pub_co_occurrences)
        self.log_info("Saved co-occurring instances to tsv")

        pub_co_occurrences.to_csv(co_occurrences_file, sep="\t", index=False)
        self.log_info("Found %s co-occurring pairs in total", len(pub_co_occurrences))

        cancer_pipeline = Pipeline([
            ("FilterCancerDisease", CancerDiseaseFilter(self.cancer_ontology.get_all_doids(), "target_id")),
            ("ExtractCancerPubMedIds", pdu.extract_unique_values("articles", cancer_pair_pubmed_ids_file)),
            ("SaveCancerInstancesAsTsv",
             pdu.to_csv(cancer_pair_instances_file, columns=["source_id", "target_id", "articles_str"]))
        ])

        cancer_pipeline.fit_transform(pub_co_occurrences)

    def create_pubmed_id_to_entity_map(self, overview_data: DataFrame) -> DataFrame:
        pubmed_to_entity_map = dict()

        for id, row in tqdm(overview_data.iterrows(), total=len(overview_data)):
            pubmed_articles = row["articles"]
            for pubmed_id in pubmed_articles:
                if pubmed_id not in pubmed_to_entity_map:
                    pubmed_to_entity_map[pubmed_id] = {"entity_ids": set() }

                pubmed_to_entity_map[pubmed_id]["entity_ids"].add(id)

        return pd.DataFrame.from_dict(pubmed_to_entity_map, orient="index")

    def find_entity_cooccurrences(self, pubmed_to_entity1: DataFrame, pubmed_to_entity2: DataFrame) -> DataFrame:
        pair_mapping = dict()
        for pubmed_id, row in tqdm(pubmed_to_entity1.iterrows(), total=len(pubmed_to_entity1)):
            if pubmed_id not in pubmed_to_entity2.index:
                continue

            entity1_ids = row["entity_ids"]
            entity2_ids = pubmed_to_entity2.loc[pubmed_id]["entity_ids"]

            entity_pairs = [str(e1) + "#" + str(e2) for e1 in entity1_ids for e2 in entity2_ids]
            for pair in entity_pairs:
                if pair not in pair_mapping:
                    id_parts = pair.split("#")
                    pair_mapping[pair] = {"id1": id_parts[0], "id2": id_parts[1], "articles": set()}

                pair_mapping[pair]["articles"].add(pubmed_id)

        co_occurence_mapping = pd.DataFrame.from_dict(pair_mapping, orient="index")

        return co_occurence_mapping

    @staticmethod
    def set_to_string(values):
        if len(values) == 0:
            return None

        return ";;;".join([str(value) for value in sorted(values)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", help="Path to the output / working directory")
    args = parser.parse_args()

    pub_preparation = PubtatorPreparation()
    pub_preparation.run(args.working_dir)
