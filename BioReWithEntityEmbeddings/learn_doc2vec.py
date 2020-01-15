import argparse
import json
import os

import gensim.models.doc2vec
import pandas as pd
import shutil

from datetime import datetime
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from io import IOBase
from tqdm import tqdm
from typing import Dict, List, Union

from utils.log_utils import LoggingMixin


class Doc2VecLearner(LoggingMixin):

    def __init__(self):
        LoggingMixin.__init__(self)

    def read_tagged_documents(self, csv_file: str, tag_splitter: str = ",") -> List[TaggedDocument]:
        self.log_info("Start reading tagged documents from %s", csv_file)
        csv_data = pd.read_csv(csv_file, "\t", encoding="utf-8")

        tagged_documents = []
        for tuple in tqdm(csv_data.itertuples(), desc="build-tagged-documents", total=len(csv_data)):
            tags = str(tuple.tags).split(tag_splitter)
            words = str(tuple.text).split(" ")
            tagged_documents.append(TaggedDocument(words, tags))

        self.log_info("Found %s tagged documents in data", len(tagged_documents))
        return tagged_documents

    def train_model(self, tagged_documents: List[TaggedDocument], configuration: Dict) -> Doc2Vec:
        self.log_info("Starting model training")

        doc2vec_config = configuration["doc2vec_config"]
        model = gensim.models.Doc2Vec(**doc2vec_config)

        self.log_info("Building vocabulary from %s documents", len(tagged_documents))
        model.build_vocab(tagged_documents)
        self.log_info("Finished creation of vocabulary. Found %s terms", len(model.wv.vocab))

        for iteration in range(configuration["iterations"]):
            self.log_info("Starting training iteration %s", (iteration + 1))
            model.train(tagged_documents, total_examples=len(tagged_documents), epochs=model.iter)

            # Reduce learning rate for the next iteration
            if configuration["adapt_learning_rate"]:
                model.alpha -= configuration["learning_rate_decay"]
                model.min_alpha = model.alpha

        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=False)

        self.log_info("Finished model training")
        return model

    def save_doc2vec_model(self, model: Doc2Vec, configuration: Dict):
        output_directory = configuration["output_directory"]
        model_name = configuration["model_name"]

        self.log_info("Start saving model %s in %s", model_name, output_directory)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        timestamp = configuration["timestamp"]
        model_file = os.path.join(output_directory, "{}_{}.doc2vec".format(timestamp, model_name))

        model.save(model_file)
        self.log_info("Succesfully saved model to %s", model_file)

        # Save a copy of the configuration file
        config_file = configuration["config_file"]
        file_name = self.get_file_name(config_file).replace(".json", "")
        target_file  = os.path.join(output_directory, "{}_{}.config.json".format(timestamp, file_name))
        shutil.copyfile(config_file, target_file)

    def get_file_name(self, file_or_path: Union[str, IOBase]) -> str:
        path = self.get_path(file_or_path)
        last_index = path.rfind(os.path.sep)

        return path[last_index + 1:] if last_index > 0 else path

    def get_path(self, file_or_path: Union[str, IOBase]) -> str:
        if isinstance(file_or_path, str):
            return file_or_path
        elif isinstance(file_or_path, IOBase):
            return file_or_path.name
        else:
            raise AssertionError("Unknown type!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doc2Vec learner")
    parser.add_argument("input_file", type=str,
                        help="Path to an tsv file containing the tagged documents")
    parser.add_argument("config_file", type=str,
                        help="Path to the training configuration (json) file")
    parser.add_argument("output_dir", type=str,
                        help="Path to the output directory")
    parser.add_argument("model_name", type=str,
                        help="Name of model to be trained")

    arguments = parser.parse_args()

    # Load and extend training configuration
    configuration = json.load(open(arguments.config_file, "r", encoding="utf-8"))
    configuration["model_name"] = arguments.model_name
    configuration["config_file"] = arguments.config_file
    configuration["output_directory"] = arguments.output_dir
    configuration["timestamp"] = datetime.now().strftime("%Y%m%d%H%M%s")

    # Load data and train model
    doc2vec_learner = Doc2VecLearner()
    tagged_documents = doc2vec_learner.read_tagged_documents(arguments.input_file)
    model = doc2vec_learner.train_model(tagged_documents, configuration)

    # Save model
    doc2vec_learner.save_doc2vec_model(model, configuration)




