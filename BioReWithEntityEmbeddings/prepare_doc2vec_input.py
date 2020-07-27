import argparse
import pandas as pd

from tqdm import tqdm
from data.pubtator import Pubtator
from utils.log_utils import LoggingMixin


class Doc2VecPreparation(LoggingMixin):

    def __init__(self):
        super(Doc2VecPreparation, self).__init__()

    def run(self, input_file: str, id_column: str, article_column: str, article_file: str, output_file: str) -> None:
        self.log_info("Reading data from %s", input_file)
        input_data = pd.read_csv(input_file, delimiter="\t", encoding="utf-8")
        self.log_info("Found %s input lines in total", len(input_data))

        pubtator = Pubtator()

        self.log_info("Reading PubTator articles from %s", article_file)
        raw_documents = pubtator.read_raw_documents_from_offsets(article_file)
        self.log_info("Found %s raw documents in total", len(raw_documents))

        self.log_info("Parsing raw documents")
        documents = pubtator.parse_raw_documents_parallel(raw_documents, 16, 10000)
        self.log_info("Finished document parsing")

        id_columns = id_column.split(" ")

        self.log_info("Start writing output")
        with open(output_file, "w", encoding="utf-8") as output_writer:
            output_writer.write("tags\ttext\n")

            for i, row in tqdm(input_data.iterrows(), desc="build-output", total=len(input_data)):
                entry_id = ";;;".join([str(row[col]) for col in id_columns])
                pubmed_ids = str(row[article_column]).split(";;;")

                complete_text = " ".join([documents[id].text() for id in pubmed_ids if id in documents])
                output_writer.write("\t".join([str(entry_id), complete_text]) + "\n")

            output_writer.close()

        self.log_info("Finished preparation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Doc2VecPreparator")
    parser.add_argument("input_file", help="Path to the input (overview) file")
    parser.add_argument("id_column", help="Names of the id columns")
    parser.add_argument("article_column", help="Name of the column with the PubMed article ids")
    parser.add_argument("article_file", help="Path to the file containing the PubMed articles in pubtator format")
    parser.add_argument("output_file", help="Path to the output file")

    args = parser.parse_args()

    extractor = Doc2VecPreparation()
    extractor.run(args.input_file, args.id_column, args.article_column, args.article_file, args.output_file)

