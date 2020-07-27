import argparse
import os

import pandas as pd
from tqdm import tqdm

from utils.log_utils import LoggingMixin


class CooccurrencePreparator(LoggingMixin):

    def __init__(self):
        super(CooccurrencePreparator, self).__init__()

    def run(self, working_directory: str, source_type: str, target_type: str):
        self.log_info(f"Start preparing {source_type}-{target_type} co-occurrences")

        source_mapping = self.read_pubmed2entity(working_directory, source_type)
        target_mapping = self.read_pubmed2entity(working_directory, target_type)

        join_result = source_mapping.join(target_mapping, lsuffix="_source", rsuffix="_target")
        join_result = join_result[join_result["entity_ids_str_target"].notnull()]
        join_result = join_result[join_result["entity_ids_str_target"].notna()]
        self.log_info(f"Found {len(join_result)} documents with co-occurrences")

        pubmed2pairs = {}
        pair2pubmed = {}

        for pubmed_id, row in tqdm(join_result.iterrows(), total=len(join_result)):
            source_ids = row["entity_ids_str_source"].split(";;;")
            target_ids = row["entity_ids_str_target"].split(";;;")

            pair_ids = [(s_id, t_id) for s_id in source_ids for t_id in target_ids]
            pubmed2pairs[str(pubmed_id)] = {
                "pair_ids_str": ";;;".join(["##".join(list(p_id)) for p_id in pair_ids])
            }

            for pair_id in pair_ids:
                if pair_id in pair2pubmed:
                    pair_entry = pair2pubmed[pair_id]
                    pair_entry["pubmed_ids_str"] += ";;;" + str(pubmed_id)
                else:
                    pair_entry = {
                        "source_id": pair_id[0],
                        "target_id": pair_id[1],
                        "pubmed_ids_str": str(pubmed_id)
                    }

                pair2pubmed[pair_id] = pair_entry

        pair_output_dir = os.path.join(working_directory, f"{source_type}-{target_type}")
        os.makedirs(pair_output_dir, exist_ok=True)

        pubmed2pair = pd.DataFrame.from_dict(pubmed2pairs, orient="index")
        pubmed2pair_file = os.path.join(pair_output_dir, "pubmed2pair.tsv")
        pubmed2pair.to_csv(pubmed2pair_file, sep="\t", index_label="pubmed_id")

        pubmed_id_file = os.path.join(pair_output_dir, "pubmed_ids.txt")
        with open(pubmed_id_file, "w") as writer:
            writer.write("\n".join([str(id) for id in pubmed2pair.index.values]))

        pair2pubmed = pd.DataFrame.from_dict(pair2pubmed, orient="index")
        pair2pubmed_file = os.path.join(pair_output_dir, "instances.tsv")
        pair2pubmed.to_csv(pair2pubmed_file, sep="\t", index=False)

    def read_pubmed2entity(self, working_directory: str, entity_type):
        input_file = os.path.join(working_directory, entity_type, "pubmed2entity.tsv")

        self.log_info(f"Reading PubMed2Entity mapping from {input_file}")
        mapping = pd.read_csv(input_file, sep="\t", index_col="pubmed_id")
        self.log_info(f"Found {len(mapping)} mapping entries")
        return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_dir", help="Path to the output / working directory")
    parser.add_argument("source_type", help="Type of the source entity (drug, disease, or mutation)")
    parser.add_argument("target_type", help="Type of the target entity (drug, disease, or mutation)")
    args = parser.parse_args()

    pub_preparation = CooccurrencePreparator()
    pub_preparation.run(args.working_dir, args.source_type, args.target_type)
