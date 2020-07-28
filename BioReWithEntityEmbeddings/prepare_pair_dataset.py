import argparse
import logging

from pathlib import Path

from data.resources import PUBTATOR_OFFSET_FILE
from extract_articles import PubtatorArticleExtractor
from prepare_doc2vec_input import Doc2VecPreparation
from pubtator_preparation import CooccurrencePreparator
from utils.log_utils import LogUtil

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
        pub_preparation.run(args.working_dir, args.source_type, args.target_type)

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
