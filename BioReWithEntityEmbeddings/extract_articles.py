import argparse
import re

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from tqdm import tqdm

from data.pubtator import PubtatorCentral
from data.resource_handler import ResourceHandler
from utils.log_utils import LoggingMixin


class PubtatorArticleExtractor(LoggingMixin):

    def __init__(self):
        super(PubtatorArticleExtractor, self).__init__()

    def run(self, offset_file: Path, pubmed_ids_file: Path, output_file: Path, threads: int, batch_size: int):
        self.log_info(f"Start extraction of PubTator documents from {offset_file}")

        self.log_info(f"Reading PubMed identifiers from {pubmed_ids_file}")
        with open(str(pubmed_ids_file), "r", encoding="utf-8") as pubmed_id_reader:
            pubmed_ids = dict([(line.strip(), 1) for line in pubmed_id_reader.readlines()])
            pubmed_id_reader.close()

        self.log_info(f"Found {len(pubmed_ids)} distinct PubMed identifiers")

        self.log_info(f"Start reading documents from {offset_file}")
        documents = PubtatorCentral().read_plain_documents(offset_file)
        self.log_info("Found %s documents in total", len(documents))

        self.logger.info(f"Creating article extraction jobs (threads={threads} | batch-size={batch_size})")
        with ThreadPoolExecutor(max_workers=threads) as executor:
            num_batches = (len(documents) - 1) // batch_size + 1

            futures = []
            for i in tqdm(range(0, len(documents), batch_size), desc="build-tasks", total=num_batches):
                document_batch = documents[i:i + batch_size]
                future = executor.submit(self.filter_documents_by_pubmed_ids, document_batch, pubmed_ids)
                futures.append(future)

            self.log_info("Submitted all tasks!")

            with open(str(output_file), "w", encoding="utf-8") as output_writer:
                for future in tqdm(futures, desc="collect-result", total=len(futures)):
                    for document in future.result():
                        output_writer.write("{}".format(document))

                output_writer.close()
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
    parser = argparse.ArgumentParser(prog="PubTator article extractor")
    parser.add_argument("--pubmed_id_file", type=str, required=True,
                        help="Path to the file containing a list of pubmed ids")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file")

    # Optional parameters
    parser.add_argument("--resource_dir", type=str, required=False, default="_resources",
                        help="Path to the directory containing the resources")
    parser.add_argument("--threads", type=int, required=False, default=16,
                        help="Number of threads to use",)
    parser.add_argument("--batch_size", type=int, required=False, default=2000,
                        help="Number of documents per job")
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    resources = ResourceHandler(resource_dir)

    extractor = PubtatorArticleExtractor()
    extractor.run(
        offset_file=resources.get_pubtator_offset_file(),
        pubmed_ids_file=Path(args.pubmed_id_file),
        output_file=Path(args.output_file),
        threads=args.threads,
        batch_size=args.batch_size
    )
