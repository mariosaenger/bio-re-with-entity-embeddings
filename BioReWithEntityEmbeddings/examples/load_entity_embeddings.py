from gensim.models import Doc2Vec

# Loading pre-trained entity model
model = Doc2Vec.load("_out/mutation/mutation-v0500.embs")

# Print number of distinct entities of the model
print(f"Distinct entities: {len(model.docvecs.doctags)}\n")

# Get the embedding for an specific entity
entity_embedding = model.docvecs["rs113488022"]
print(f"Embedding of rs113488022:\n{entity_embedding}\n")

# Find similar entities
print("Most similar entities to rs113488022:")
top5_nearest_neighbors = model.docvecs.most_similar("rs113488022", topn=5)
for i, (entity_id, sim) in enumerate(top5_nearest_neighbors):
    print(f" {i+1}: {entity_id} (similarity: {sim:.3f})")
