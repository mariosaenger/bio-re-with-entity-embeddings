import argparse
import logging

from pathlib import Path

from data.resources import PUBTATOR_OFFSET_FILE
from extract_articles import PubtatorArticleExtractor
from prepare_doc2vec_input import Doc2VecPreparation
from pubtator_preparation import PubtatorMutationDataSetPreparation, EntityDataSetPreparation, \
    PubtatorDrugOccurrencesPreparation, PubtatorDiseaseOccurrencesPreparation
from utils.log_utils import LogUtil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, required=True,
                        help="Directory which will hold all (intermediate) output files")
    parser.add_argument("--entity_type", type=str, required=True,
                        choices=["drug", "disease", "mutation"],
                        help="Target entity type (drug, disease, or mutation)")

    parser.add_argument("--use_caching", type=bool, required=False, default=True,
                        help="Indicate whether to perform all preparation steps from scratch or use "
                             "existing results from previous executions.")

    args = parser.parse_args()

    logger = LogUtil.create_logger("prepare_pair_dataset", logging.INFO)
    logger.info(f"Start preparation of {args.entity_type} data set")

    use_caching = args.use_caching

    working_dir = Path(args.working_dir)
    entity_type_dir = working_dir / args.entity_type

    instance_file = entity_type_dir / "instances.tsv"
    pubmed_ids_file = entity_type_dir / "pubmed_ids.txt"
    pubmed2entity_file = entity_type_dir / "pubmed2entity.tsv"

    # Run preparation of the data set for the given entity type using the information
    # provided by the PubTator files
    if (
            use_caching is False
            or
            not (
                instance_file.exists() and
                pubmed_ids_file.exists() and
                pubmed2entity_file.exists()
            )
    ):
        entity_ds_preparator: EntityDataSetPreparation = None
        if args.entity_type == "mutation":
            entity_ds_preparator = PubtatorMutationDataSetPreparation()
        elif args.entity_type == "drug":
            entity_ds_preparator = PubtatorDrugOccurrencesPreparation()
        elif args.entity_type == "disease":
            entity_ds_preparator = PubtatorDiseaseOccurrencesPreparation()
        else:
            raise NotImplementedError(f"Unsupported entity type {args.entity_type}")

        entity_ds_preparator.run(
            working_dir=working_dir
        )

        # Skipping caching for all following steps to prevent unintended caching issues
        use_caching = False

    else:
        logger.info("Skipping PubTator preparation")

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
            id_columns=["entity_id"],
            article_file=articles_file,
            article_column="articles_str",
            output_file=doc2vec_input_file
        )

    else:
        logger.info("Skipping preparation of the Doc2Vec input")

    logger.info(f"Finished preparation of {args.entity_type} data set")
