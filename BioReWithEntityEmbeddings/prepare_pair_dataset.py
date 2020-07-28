import argparse
import logging
import pandas as pd

from pathlib import Path

from tqdm import tqdm

from data.resources import PUBTATOR_OFFSET_FILE
from extract_articles import PubtatorArticleExtractor
from prepare_doc2vec_input import Doc2VecPreparation
from utils.log_utils import LogUtil, LoggingMixin


class CooccurrencePreparator(LoggingMixin):

    def __init__(self):
        super(CooccurrencePreparator, self).__init__()

    def run(self, working_directory: Path, source_type: str, target_type: str):
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
                    pair_entry["articles_str"] += ";;;" + str(pubmed_id)
                else:
                    pair_entry = {
                        "source_id": pair_id[0],
                        "target_id": pair_id[1],
                        "articles_str": str(pubmed_id)
                    }

                pair2pubmed[pair_id] = pair_entry

        pair_output_dir = working_directory / f"{source_type}-{target_type}"
        pair_output_dir.mkdir(parents=True, exist_ok=True)

        pubmed2pair = pd.DataFrame.from_dict(pubmed2pairs, orient="index")
        pubmed2pair_file = pair_output_dir / "pubmed2pair.tsv"
        pubmed2pair.to_csv(pubmed2pair_file, sep="\t", index_label="pubmed_id")

        pubmed_id_file = pair_output_dir / "pubmed_ids.txt"
        with open(str(pubmed_id_file), "w") as writer:
            writer.write("\n".join([str(id) for id in pubmed2pair.index.values]))

        pair2pubmed = pd.DataFrame.from_dict(pair2pubmed, orient="index")
        pair2pubmed_file = pair_output_dir / "instances.tsv"
        pair2pubmed.to_csv(pair2pubmed_file, sep="\t", index=False)

    def read_pubmed2entity(self, working_directory: Path, entity_type) -> pd.DataFrame:
        input_file = working_directory / entity_type / "pubmed2entity.tsv"

        self.log_info(f"Reading PubMed2Entity mapping from {input_file}")
        mapping = pd.read_csv(input_file, sep="\t", index_col="pubmed_id")
        self.log_info(f"Found {len(mapping)} mapping entries")

        return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, required=True,
                        help="Path to the output / working directory")
    parser.add_argument("--source_type", type=str, required=True,
                        choices=["drug", "disease", "mutation"],
                        help="Type of the source entity (drug, disease, or mutation)")
    parser.add_argument("--target_type", type=str, required=True,
                        choices=["drug", "disease", "mutation"],
                        help="Type of the target entity (drug, disease, or mutation)")

    # Optional arguments
    parser.add_argument("--use_caching", type=bool, required=False, default=True,
                        help="Indicate whether to perform all preparation steps from scratch or use "
                             "existing results from previous executions.")

    args = parser.parse_args()

    logger = LogUtil.create_logger("prepare_pair_dataset", logging.INFO)
    logger.info(f"Start preparation of {args.source_type}-{args.target_type} data set")

    use_caching = args.use_caching

    working_dir = Path(args.working_dir)

    source_entity_dir = working_dir / args.source_type
    if not source_entity_dir.is_dir():
        logger.error(f"Can't find directory {source_entity_dir}. "
                     f"Please run preparation of {args.source_type} annotations first")
        exit()

    target_entity_dir = working_dir / args.source_type
    if not target_entity_dir.is_dir():
        logger.error(f"Can't find directory {target_entity_dir}. "
                    f"Please run preparation of {args.target_type} annotations first")
        exit()

    entity_type_dir = working_dir / f"{args.source_type}-{args.target_type}"

    instance_file = entity_type_dir / "instances.tsv"
    pubmed_ids_file = entity_type_dir / "pubmed_ids.txt"
    pubmed2pair_file = entity_type_dir / "pubmed2par.tsv"

    # Run preparation of the data set for the given entity types using the information
    # provided by the PubTator files
    if (
            use_caching is False
            or
            not (
                    instance_file.exists() and
                    pubmed_ids_file.exists() and
                    pubmed2pair_file.exists()
            )
    ):
        pub_preparation = CooccurrencePreparator()
        pub_preparation.run(
            working_directory=working_dir,
            source_type=args.source_type,
            target_type=args.target_type
        )

        # Skipping caching for all following steps to prevent unintended caching issues
        use_caching = False

    else:
        logger.info("Skipping co-occurrence preparation")

    # Extract the necessary articles from the PubTator offset file
    articles_file = entity_type_dir / "articles.txt"
    if (
            use_caching is False
            or
            not articles_file.exists()
    ):
        extractor = PubtatorArticleExtractor()
        extractor.run(
            offset_file=PUBTATOR_OFFSET_FILE,
            pubmed_ids_file=pubmed_ids_file,
            output_file=articles_file,
            threads=16,
            batch_size=2000
        )

        # Skipping caching for all following steps to prevent unintended caching issues
        use_caching = False
    else:
        logger.info("Skipping extraction of articles")

    # Create the input files for the Doc2Vec entity embedding learning
    doc2vec_input_file = entity_type_dir / "doc2vec_input.txt"
    if (
        use_caching is False
        or
        not doc2vec_input_file.exists()
    ):
        extractor = Doc2VecPreparation()
        extractor.run(
            input_file=instance_file,
            id_columns=["source_id", "target_id"],
            article_file=articles_file,
            article_column="articles_str",
            output_file=doc2vec_input_file
        )
    else:
        logger.info("Skipping preparation of the Doc2Vec input")

    logger.info(f"Finished preparation of {args.source_type}-{args.target_type} data set")
