import argparse
import re

from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm

from data.pubtator import Pubtator
from utils.log_utils import LoggingMixin


class PubtatorArticleExtractor(LoggingMixin):

    def __init__(self):
        super(PubtatorArticleExtractor, self).__init__()

    def run(self, offset_file: str, pubmed_ids_file: str, output_file: str, threads: int, batch_size: int):
        self.log_info("Start extraction of pubtator documents from %s", offset_file)
        self.log_info("Reading pubmed ids from %s", pubmed_ids_file)

        with open(pubmed_ids_file, "r", encoding="utf-8") as pubmed_id_reader:
            pubmed_ids = dict([(line.strip(), 1) for line in pubmed_id_reader.readlines()])
            pubmed_id_reader.close()

        self.log_info("Found %s distinct pubmed ids", len(pubmed_ids))

        self.log_info("Start reading documents from %s", offset_file)
        documents = Pubtator().read_raw_documents_from_offsets(offset_file)
        self.log_info("Found %s documents in total", len(documents))

        self.logger.info("Creating article extraction jobs (%s threads and %s docs / thread)", threads, batch_size)
        with ThreadPoolExecutor(max_workers=threads) as executor:
            num_batches = len(documents) / batch_size

            futures = []
            for i in tqdm(range(0, len(documents), batch_size), desc="build-tasks", total=num_batches):
                document_batch = documents[i:i + batch_size]
                future = executor.submit(self.filter_documents_by_pubmed_ids, document_batch, pubmed_ids)
                futures.append(future)

            self.log_info("Submitted all tasks!")

            with open(output_file, "w", encoding="utf-8") as output_writer:
                for future in tqdm(futures, desc="collect-result", total=len(futures)):
                    for document in future.result():
                        output_writer.write("{}".format(document))

                output_writer.close()

            self.log_info("Shutting down thread pool executor")
            executor.shutdown()

        self.log_info("Finished extraction of documents")


    def filter_documents_by_pubmed_ids(self, raw_documents: List[str], pubmed_ids: dict) -> List[str]:
        title_regex = re.compile("([0-9]+)\\|t\\|(.*)")
        abstract_regex = re.compile("([0-9]+)\\|a\\|(.*)")

        matching_documents = list()
        for raw_document in raw_documents:
            document_lines = raw_document.splitlines(False)
            pubmed_id = None

            for line in document_lines:
                title_match = title_regex.match(line)
                if title_match:
                    pubmed_id = title_match.group(1).strip()
                    break

                abstract_match = abstract_regex.match(line)
                if abstract_match:
                    pubmed_id = abstract_match.group(1).strip()
                    break

            if pubmed_id is not None and pubmed_id in pubmed_ids:
                matching_documents.append(raw_document)

        return matching_documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Pubtator article extractor")
    parser.add_argument("offset_file", help="Path to the input pubtator offsets file")
    parser.add_argument("pubmed_id_list", help="Path to the file containing a list of pubmed ids")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("threads", help="Number of threads to use", type=int)
    parser.add_argument("batch_size", help="Number of documents per job", type=int)

    args = parser.parse_args()

    extractor = PubtatorArticleExtractor()
    extractor.run(args.offset_file, args.pubmed_id_list, args.output_file, args.threads, args.batch_size)
