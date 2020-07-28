import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from typing import List

from data.pubtator import Pubtator
from utils.log_utils import LoggingMixin


class Doc2VecPreparation(LoggingMixin):

    def __init__(self):
        super(Doc2VecPreparation, self).__init__()

    def run(self,
            input_file: Path,
            id_columns: List[str],
            article_file: Path,
            article_column: str,
            output_file: Path):

        self.log_info("Starting Doc2Vec preparation")

        self.log_info(f"Reading data from {input_file}")
        input_data = pd.read_csv(input_file, delimiter="\t", encoding="utf-8")
        self.log_info(f"Found {len(input_data)} input lines in total")

        pubtator = Pubtator()

        self.log_info(f"Reading PubTator articles from {article_file}")
        raw_documents = pubtator.read_raw_documents_from_offsets(article_file)
        self.log_info(f"Found {len(raw_documents)} raw documents in total")

        self.log_info("Parsing raw documents")
        #FIXME: Make number of threads and batch size parameterizable
        documents = pubtator.parse_raw_documents_parallel(raw_documents, 16, 10000)
        self.log_info("Finished document parsing")

        self.log_info("Start writing output")
        with open(str(output_file), "w", encoding="utf-8") as output_writer:
            output_writer.write("tags\ttext\n")

            for i, row in tqdm(input_data.iterrows(), desc="build-output", total=len(input_data)):
                entry_id = ";;;".join([str(row[col]) for col in id_columns])
                pubmed_ids = str(row[article_column]).split(";;;")

                complete_text = " ".join([documents[id].text() for id in pubmed_ids if id in documents])
                output_writer.write("\t".join([str(entry_id), complete_text]) + "\n")

            output_writer.close()

        self.log_info("Finished Doc2Vec preparation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Doc2VecPreparator")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input (overview) file")
    parser.add_argument("--id_column", type=str, required=True,
                        help="Names of the id columns (separated by whitespace)")
    parser.add_argument("--article_file", type=str, required=True,
                        help="Path to the file containing the PubMed articles in pubtator format")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file")

    parser.add_argument("--article_column", type=str, required=False, default="articles_str",
                        help="Name of the column with the PubMed article ids (separated by ;;;")

    args = parser.parse_args()

    extractor = Doc2VecPreparation()
    extractor.run(
        input_file=Path(args.input_file),
        id_columns=args.id_column.split(" "),
        article_file=Path(args.article_file),
        article_column=args.article_column,
        output_file=Path(args.output_file)
    )

