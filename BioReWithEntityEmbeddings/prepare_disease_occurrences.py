import argparse
import os

from pandas import DataFrame
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from typing import Callable

from data.disease_ontology import DiseaseOntology
from data.pubtator import Pubtator, PubtatorPreparationUtils
from data.resources import PUBTATOR_DIS2PUB_FILE, DO_ONTOLOGY_FILE
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PipelineMixin
from utils.pandas_utils import PandasUtil as pdu


class MeshTermToDoidMapper(PipelineMixin):

    def __init__(self, disease_ontology: DiseaseOntology, mesh_extractor: Callable, source_id_extractor: Callable):
        super(MeshTermToDoidMapper, self).__init__()
        self.disease_ontology = disease_ontology
        self.mesh_extractor = mesh_extractor
        self.source_id_extractor = source_id_extractor

    def transform(self, data: DataFrame, y=None):
        self.log_info("Adding DOID id to %s instances", len(data))
        num_unknown_doid = 0

        new_data_map = dict()
        for id, row in tqdm(data.iterrows(), total=len(data)):
            disease_mesh_term = self.mesh_extractor(id, row)

            doids = self.disease_ontology.get_doid_by_mesh(disease_mesh_term)
            if len(doids)  > 0:
                source_id = self.source_id_extractor(id, row)
                for doid in doids:
                    row_copy = row.copy()
                    row_copy["doid"] = doid

                    new_row_id = source_id + "#" + doid
                    row_copy["id_doid"] = new_row_id

                    if new_row_id in new_data_map:
                        a1 = row_copy["articles"]
                        a2 = row_copy["articles"]
                        row_copy["articles"] = a1.union(a2)

                    new_data_map[new_row_id] = row_copy
            else:
                num_unknown_doid = num_unknown_doid + 1

        new_data = DataFrame(list(new_data_map.values()))
        new_data.index = new_data["id_doid"]
        new_data = new_data.drop("id_doid", axis=1)

        self.log_info("Can't find DOID for %s of %s entries", num_unknown_doid, len(data))
        self.log_info("Finished MeSH to DOID mapping. New data set has %s instances", len(new_data))

        return new_data


class DrugOccurrencesPreparation(LoggingMixin):

    def __init__(self):
        super(DrugOccurrencesPreparation, self).__init__()

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
        mesh_extractor = lambda id, row: id
        id_extractor = lambda id, row: id

        disease_pipeline = Pipeline([
            ("MapMashTermTodoDoid", MeshTermToDoidMapper(self.disease_ontology, mesh_extractor, id_extractor)),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", help="Path to the output / working directory")
    args = parser.parse_args()

    pub_preparation = DrugOccurrencesPreparation()
    pub_preparation.run(args.working_dir)
