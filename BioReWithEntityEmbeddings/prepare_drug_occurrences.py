import argparse
import os
import pandas as pd

from sklearn.pipeline import Pipeline
from tqdm import tqdm
from typing import List

from data.pubtator import Pubtator, PubtatorPreparationUtils
from data.resources import PUBTATOR_CHEM2PUB_FILE, MESH_TO_DRUGBANK_MAPPING
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu, PipelineMixin


class MeshTermToDrugBankIdMapper(PipelineMixin):

    def __init__(self, mesh_to_drugbank: pd.DataFrame):
        super(MeshTermToDrugBankIdMapper, self).__init__()
        self.mesh_to_drugbank = mesh_to_drugbank

    def transform(self, data: pd.DataFrame, y=None):
        self.log_info(f"Adding DrugBank identifier to {len(data)} instances")

        new_data_map = dict()
        for mesh, row in tqdm(data.iterrows(), total=len(data)):
            drugbank_ids = self.get_drugbank_ids_by_mesh(mesh)

            if len(drugbank_ids) > 0:
                for db_id in drugbank_ids:
                    row_copy = row.copy()
                    row_copy["drugbank_id"] = db_id

                    if db_id in new_data_map:
                        a1 = row_copy["articles"]
                        a2 = new_data_map[db_id]["articles"]
                        row_copy["articles"] = a1.union(a2)

                    new_data_map[db_id] = row_copy

        new_data = pd.DataFrame(list(new_data_map.values()))
        new_data.index = new_data["drugbank_id"]
        new_data = new_data.drop("mesh_db_id", axis=1)

        self.log_info("Finished MeSH to DrugBank id mapping. New data set has %s instances", len(new_data))

        return new_data

    def get_drugbank_ids_by_mesh(self, mesh: str) -> List[str]:
        if mesh in self.mesh_to_drugbank.index:
            return self.mesh_to_drugbank.loc[mesh]["DrugBankIDs"].split("|")

        return []


class DrugOccurrencesPreparation(LoggingMixin):

    def __init__(self):
        super(DrugOccurrencesPreparation, self).__init__()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", help="Path to the output / working directory")
    args = parser.parse_args()

    pub_preparation = DrugOccurrencesPreparation()
    pub_preparation.run(args.working_dir)
