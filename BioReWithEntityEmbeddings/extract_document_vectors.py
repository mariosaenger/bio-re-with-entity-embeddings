import argparse

from pathlib import Path
from gensim.models import Doc2Vec

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Helper script to extract the document vectors from gensim Doc2Vec models")
    parser.add_argument("--embedding_model", type=Path, required=True,
                        help="Path to the embedding model to be converted")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Path to the output directory")
    args = parser.parse_args()

    model_file:Path = args.embedding_model
    output_dir:Path = args.output_dir

    if not (model_file.exists() and model_file.is_file()):
        print(f"Embedding model {model_file} isn't a valid file")
        exit()

    print(f"Loading model from {model_file}")
    model = Doc2Vec.load(str(model_file))

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{model_file.stem}.bin"
    print(f"Saving document vectors to {output_file}")

    # Save only the document vectors of the model
    # (limit file to be less than < 2GB, otherwise doc vectors will be split in separate files)
    model.docvecs.save(str(output_file), sep_limit=1024 * 1024 * 1024 * 3)

    vocab_file = output_dir / f"{model_file.stem}.vocab"
    print(f"Saving vocabulary to {vocab_file}")

    with open(str(vocab_file), "w") as writer:
        writer.write("\n".join(model.docvecs.doctags.keys()))



