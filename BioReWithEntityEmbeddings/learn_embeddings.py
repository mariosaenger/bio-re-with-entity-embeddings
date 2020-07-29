import argparse
import json

import gensim.models.doc2vec
import pandas as pd
import shutil

from datetime import datetime
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from utils.log_utils import LoggingMixin


class Doc2VecEmbeddingLearner(LoggingMixin):

    def __init__(self):
        LoggingMixin.__init__(self)

    def read_tagged_documents(self, csv_file: Path, tag_splitter: str = ",") -> List[TaggedDocument]:
        self.log_info(f"Start reading tagged documents from {csv_file}")
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
            self.log_info(f"Starting training iteration {iteration + 1}")
            model.train(tagged_documents, total_examples=len(tagged_documents), epochs=1)

            # Reduce learning rate for the next iteration
            if configuration["adapt_learning_rate"]:
                model.alpha -= configuration["learning_rate_decay"]
                model.min_alpha = model.alpha

        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=False)

        self.log_info("Finished model training")
        return model

    def save_doc2vec_model(self, model: Doc2Vec, configuration: Dict):
        output_directory = configuration["output_directory"]
        output_directory.mkdir(parents=True, exist_ok=True)

        model_name = configuration["model_name"]
        model_file = output_directory / f"{model_name}.embs"

        self.log_info(f"Start saving model {model_name} to {output_directory}")
        model.save(str(model_file))
        self.log_info(f"Saved model to {model_file}")

        self.log_info(f"Saving vocabulary")
        vocab_file = output_directory / f"{model_name}.vocab"
        with open(str(vocab_file), "w") as writer:
            writer.write("\n".join(model.docvecs.doctags.keys()))

        # Save a copy of the configuration file
        self.log_info(f"Saving configuration")
        config_file = configuration["config_file"]
        target_file = output_directory / f"{model_name}_config.json"
        shutil.copyfile(str(config_file), str(target_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doc2Vec learner")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to an tsv file containing the tagged documents")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to the Doc2Vec configuration file (in json format)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of model to be trained")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory")

    arguments = parser.parse_args()

    # Load and extend training configuration
    configuration = json.load(open(arguments.config_file, "r", encoding="utf-8"))
    configuration["model_name"] = arguments.model_name
    configuration["config_file"] = Path(arguments.config_file)
    configuration["output_directory"] = Path(arguments.output_dir)
    configuration["timestamp"] = datetime.now().strftime("%Y%m%d%H%M%s")

    # Load data and train model
    doc2vec_learner = Doc2VecEmbeddingLearner()
    tagged_documents = doc2vec_learner.read_tagged_documents(Path(arguments.input_file))
    model = doc2vec_learner.train_model(tagged_documents, configuration)

    # Save model
    doc2vec_learner.save_doc2vec_model(model, configuration)




