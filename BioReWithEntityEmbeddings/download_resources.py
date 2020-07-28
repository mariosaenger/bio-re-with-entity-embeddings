import argparse
import re
import tempfile
import os
import shutil
import urllib.request as request

from pathlib import Path
from tqdm import tqdm

from utils.log_utils import LoggingMixin


class ResourceDownloader(LoggingMixin):

    def download_and_unzip_archive(self, archive_url: str, cache_dir: Path):
        file_name = self.get_plain_archive_name(archive_url)

        self.log_info(f"Downloading archive {archive_url}")
        file_path = cache_dir / file_name
        if file_path.exists():
            self.log_info(f"Archive {file_name} already in cache!")
            return

        cached_archive = self.cache_resource(archive_url, cache_dir)
        self.unpack_file(cached_archive, cache_dir / file_name, keep=False)

    def download_file(self, file_url: str, cache_dir: Path):
        self.cache_resource(file_url, cache_dir)

    def cache_resource(self, resource_url: str, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = re.sub(r".+/", "", resource_url)

        cache_path = cache_dir / filename
        if cache_path.exists():
            return cache_path

        fd, temp_filename = tempfile.mkstemp()
        self.log_info(f"{resource_url} not found in cache, downloading to {temp_filename}")

        # GET file object
        response = request.urlopen(resource_url)
        content_length = int(response.headers["content-length"])
        block_size = 8192

        if content_length:
            length = int(content_length)
            block_size = max(8192, length // 100)

        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total)

        with open(temp_filename, "wb") as writer:
            while True:
                block = response.read(block_size)
                if not block:
                    break
                writer.write(block)
                progress.update(len(block))

        progress.close()

        self.log_info(f"Copying {temp_filename} to cache at {cache_path}")
        shutil.copyfile(temp_filename, str(cache_path))
        self.log_info(f"Removing temp file {temp_filename}")

        os.close(fd)
        os.remove(temp_filename)

        return cache_path

    def unpack_file(self, file: Path, unpack_to: Path, mode: str = None, keep: bool = True):
        if mode == "zip" or (mode is None and str(file).endswith("zip")):
            from zipfile import ZipFile

            with ZipFile(file, "r") as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(unpack_to)

        elif mode == "targz" or (
                mode is None and str(file).endswith("tar.gz") or str(file).endswith("tgz")
        ):
            import tarfile

            with tarfile.open(file, "r:gz") as tarObj:
                tarObj.extractall(unpack_to)

        elif mode == "tar" or (mode is None and str(file).endswith("tar")):
            import tarfile

            with tarfile.open(file, "r") as tarObj:
                tarObj.extractall(unpack_to)

        elif mode == "gz" or (mode is None and str(file).endswith("gz")):
            import gzip

            with gzip.open(str(file), "rb") as f_in:
                with open(str(unpack_to), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        else:
            if mode is None:
                raise AssertionError(f"Can't infer archive type from {file}")
            else:
                raise AssertionError(f"Unsupported mode {mode}")

        if not keep:
            os.remove(str(file))

    def get_plain_archive_name(self, archive_url: str) -> str:
        file_name = re.sub(r".+/", "", archive_url)

        if file_name.endswith(".tar.gz"):
            file_name = file_name.replace(".tar.gz", "")
        elif file_name.endswith(".gz"):
            file_name = file_name.replace(".gz", "")
        elif file_name.endswith(".zip"):
            file_name = file_name.replace(".zip", "")
        elif file_name.endswith(".tar"):
            file_name = file_name.replace(".tar", "")

        return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", type=str, required=True, nargs="+",
                        choices=["pubtator", "pubtator_central", "disease_ontology"],
                        help="Resource to download")

    args = parser.parse_args()

    archives_to_download = []
    files_to_download = []

    if "pubtator" in args.resource:
        archives_to_download += [
            ("pubtator", "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/chemical2pubtator.gz"),
            ("pubtator", "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/disease2pubtator.gz"),
            ("pubtator", "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/gene2pubtator.gz"),
            ("pubtator", "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/species2pubtator.gz"),
            ("pubtator", "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/mutation2pubtator.gz"),

            ("pubtator", "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator/bioconcepts2pubtator_offsets.gz")
        ]

    if "pubtator_central" in args.resource:
        archives_to_download += [
            ("pubtator_central", "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz")
        ]

    if "disease_ontology" in args.resource:
        files_to_download += [
            ("do", "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.obo")
        ]

    cache_dir = Path("_cache")

    downloader = ResourceDownloader()
    for resource, archive_url in archives_to_download:
        downloader.download_and_unzip_archive(archive_url, cache_dir / resource)

    for resource, file_url in files_to_download:
        downloader.download_file(file_url, cache_dir / resource)






