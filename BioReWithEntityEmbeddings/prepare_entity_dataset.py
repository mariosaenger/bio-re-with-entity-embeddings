import argparse
from pathlib import Path

from data.resources import PUBTATOR_OFFSET_FILE
from extract_articles import PubtatorArticleExtractor
from prepare_doc2vec_input import Doc2VecPreparation
from pubtator_preparation import PubtatorMutationDataSetPreparation, EntityDataSetPreparation, \
    PubtatorDrugOccurrencesPreparation, PubtatorDiseaseOccurrencesPreparation

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

    entity_ds_preparator: EntityDataSetPreparation = None
    if args.entity_type == "mutation":
        entity_ds_preparator = PubtatorMutationDataSetPreparation()
    elif args.entity_type == "drug":
        entity_ds_preparator = PubtatorDrugOccurrencesPreparation()
    elif args.entity_type == "disease":
        entity_ds_preparator = PubtatorDiseaseOccurrencesPreparation()
    else:
        raise NotImplementedError(f"Unsupported entity type {args.entity_type}")

    #TODO: Download pubtator files automatically!

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
        entity_ds_preparator.run(args.working_dir)

        # Skipping caching for all following steps to prevent unintended caching issues
        use_caching = False

    else:
        print("Skipping PubTator preparation ...")

    # Extract the necessary articles from the PubTator offset file
    articles_file = entity_type_dir / "articles.txt"
    if (
            use_caching is False
            or
            not articles_file.exists()
    ):
        extractor = PubtatorArticleExtractor()
        extractor.run(PUBTATOR_OFFSET_FILE, str(pubmed_ids_file), str(articles_file), 16, 2000)

        # Skipping caching for all following steps to prevent unintended caching issues
        use_caching = False
    else:
        print("Skipping extraction of articles")

    # Create the input files for the Doc2Vec entity embedding learning
    doc2vec_input_file = entity_type_dir / "doc2vec_input.txt"
    if (
        use_caching is False
        or
        not doc2vec_input_file.exists()
    ):
        extractor = Doc2VecPreparation()
        extractor.run(str(instance_file), "entity_id", "articles_str", str(articles_file), str(doc2vec_input_file))

    else:
        print("Skipping preparation of the Doc2Vec input")

    print("Finished preparation")
