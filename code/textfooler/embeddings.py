import numpy as np

# This file uses embeddings, in the form of dictionaries between strings and numpy arrays,
#   to perform semantic similarity functions, synonym detection, and so forth.

# Helpful wrapper for numpy routines for cosine similarity
def cosineSim(vec1,vec2):
  return np.inner(vec1,vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
cosine_sim = cosineSim
cosine_similarity = cosineSim

# Return similarity between two words, given an embedding.
def word_similarity_embedding(word1,word2,embedding):
  if word1 in embedding and word2 in embedding:
    return cosineSim(embedding[word1],embedding[word2])
  else:
    if word1 not in embedding:
      print(word1,"is not in the provided embedding. Returning 0 similarity.")
      return 0.0
    else:
      print(word2,"is not in the provided embedding. Returning 0 similarity.")
      return 0.0

# Construct a sentence vector using its component word vectors, given an embedding.
# Returns None if no words were in the embedding, otherwise averages together the word vectors.
def sentence_vector(sentence, embedding):
  words = sentence.split()
  vec = None
  numwords = 0.0
  for word in words:
    # Only add words the embedding contains
    if word in embedding:
      # Add the word vector and increment our denominator
      if numwords > 0:
        vec = vec + embedding[word]
      else:
        vec = embedding[word]
      numwords = numwords + 1.0
    else:
      # Uncomment for debugging
      # print("Unrecognized word in sentence, skipping:",word)
      continue
  if numwords > 0:
    vec = vec / float(numwords)
    return vec
  else:
    print("No words in sentence were contained in the embedding.")
    print(sentence)
    return None

# Return similarity between two sentences, given an embedding.
def semantic_sim_embedding(sentence1, sentence2, embedding):
  se1 = sentence_vector(sentence1,embedding)
  se2 = sentence_vector(sentence2,embedding)
  if se1 and se2:
    return cosineSim(se1,se2)
  else:
    print("One or both sentences contained no words in the embedding. Returning 0 similarity.")
    return 0

# Find the most similar words to a given word, given an embedding.
def synonym_embedding(word,embedding,num=10):
  rvals = list()
  worst_score = 100 # this should get overwritten by the first word in the embedding
  # O(vocab), worst case O(2*num*vocab)
  for key in embedding.keys():
    # Don't return the word as its own synonym.
    if key == word:
      continue
    # If we still need words, just take em.
    if len(rvals) < num:
      rvals.append(key)
      # update worst score
      score = word_similarity_embedding(word,key,embedding)
      if score < worst_score:
        worst_score = score
    # Otherwise, only take words better than our worst known synonym.
    else:
      score = word_similarity_embedding(word,key,embedding)
      if score > worst_score:
        rvals.append(key)
        # Remove lowest score, and update worst score
        # There's gotta be a clever way to do this with list comprehensions but I just don't care right now
        worst_word = rvals[0]
        for rval in rvals:
          if word_similarity_embedding((word,rval,embedding) == worst_score:
            worst_word = rval
        rvals.remove(worst_word)
        worst_score = 100
        for rval in rvals:
          score = word_similarity_embedding(word,rval,embedding)
          if score < worst_score:
            worst_score = score
  return rvals

# Could easily parameterize the above function to take any arbitrary word similarity function.
