import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from typing import List

from data.pubtator import PubtatorCentral
from utils.log_utils import LoggingMixin


class Doc2VecPreparation(LoggingMixin):

    def __init__(self):
        super(Doc2VecPreparation, self).__init__()

    def run(self,
            pubmed_to_ids_file: Path,
            pubmed_id_column: str,
            ids_column: str,
            article_file: Path,
            output_file: Path):

        self.log_info("Starting Doc2Vec preparation")

        self.log_info(f"Reading PubMed-to-Ids from {pubmed_to_ids_file}")
        input_data = pd.read_csv(pubmed_to_ids_file, delimiter="\t", encoding="utf-8")
        self.log_info(f"Found {len(input_data)} input lines in total")

        pubtator = PubtatorCentral()

        self.log_info(f"Reading plain PubTator documents from {article_file}")
        raw_documents = pubtator.read_plain_documents(article_file)
        self.log_info(f"Found {len(raw_documents)} raw documents in total")

        self.log_info("Parsing raw documents")
        #FIXME: Make number of threads and batch size parameterizable
        documents = pubtator.parse_raw_documents_parallel(raw_documents, 16, 2000)
        self.log_info("Finished document parsing")

        self.log_info("Start writing output")
        with open(str(output_file), "w", encoding="utf-8") as output_writer:
            output_writer.write("tags\ttext\n")

            for i, row in tqdm(input_data.iterrows(), desc="build-output", total=len(input_data)):
                pubmed_id = str(row[pubmed_id_column])
                if pubmed_id not in documents:
                    continue

                tags = row[ids_column]
                output_writer.write("\t".join([tags, documents[pubmed_id].text()]) + "\n")

            output_writer.close()

        self.log_info("Finished Doc2Vec preparation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Doc2VecPreparator")
    parser.add_argument("--pubmed2ids_file", type=str, required=True,
                        help="Path to the input pubmed2entity or pubmed2pair file")
    parser.add_argument("--ids_columns", type=str, required=True,
                        help="Name of the column containing the entity / entity pair ids (separated with ;;;)")
    parser.add_argument("--article_file", type=str, required=True,
                        help="Path to the file containing the PubMed articles in pubtator format")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file")

    # Optional parameters
    parser.add_argument("--pubmed_id_column", type=str, required=False, default="pubmed_id",
                        help="Name of the column containing the pubmed id")

    args = parser.parse_args()

    extractor = Doc2VecPreparation()
    extractor.run(
        pubmed_to_ids_file=Path(args.pubmed2ids_file),
        pubmed_id_column=args.pubmed_id_column,
        ids_column=args.ids_columns,
        article_file=Path(args.article_file),
        output_file=Path(args.output_file)
    )

