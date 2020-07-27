import argparse
import os

from sklearn.pipeline import Pipeline

from data.pubtator import Pubtator, PubtatorPreparationUtils
from data.resources import PUBTATOR_MUT2PUB_FILE
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu


class MutationOccurrencesPreparation(LoggingMixin):

    def __init__(self):
        super(MutationOccurrencesPreparation, self).__init__()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", help="Path to the output / working directory")
    args = parser.parse_args()

    pub_preparation = MutationOccurrencesPreparation()
    pub_preparation.run(args.working_dir)
