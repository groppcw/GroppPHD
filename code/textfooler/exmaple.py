import attack
import embeddings



# Load in settings.

CLASSIFIER_FUNC = lambda x: []
EMBEDDING = dict()
# IMPORTANCE_THRESHOLD
# NUM_SYNONYMS
SIM_THRESHOLD = 0.5

# Construct necessary functions, which involves some lambda usage


# Assign classifier function. This is tightly dependent on the classifier being tested.
classifier_func = CLASSIFIER_FUNC # classifier_func(text) returns array of class probabilities

# Assign importance function. This is going to be passed a text and the above classifier function.
importance_func = attack.importance_scores
# If you want to change the OOV default from "OUT_OF_VOCABULARY", do this instead:
# importance_func = lambda x,y : attack.importance_scores(x,y,oov="OOV")
# If you want to use a non-query-based importance function, do something like:
# importance_func = lambda x,y : attack.importance_length(x)

# Assign selection function for what we do with the importance scores generated above.
select_func = attack.select_importance_threshold
# To change the default threshold:
# select_func = lambda x,y : attack.select_importance_threshold(x,y,IMPORTANCE_THRESHOLD)
# To ignore importance and just remove stopwords:
# select_func = lambda x,y : attack.select_non_stopword(x)
# To use both:
# select_func = attack.create_composite_select_w_val(attack.select_importance_threshold,lambda x,y : attack.select_non_stopword(x))

# Assign synonym generation function
synonym_func = lambda x : embeddings.synonym_embedding(x,EMBEDDING)
# If you want to use a different number of returns than the default 10, do this:
# synonym_func = lambda x : embeddings.synonym_embedding(x,EMBEDDING,NUM_SYNONYMS)

# Assign candidate filter (word-level)
candidate_word_filter = None

# Assign candidate filter (sentence-level)
candidate_sentence_filter = attack.semantic_pos_filter

# Assign semantic similarity function and threshold
sim_func = lambda x,y : embeddings.semantic_sim_embedding(x,y,EMBEDDING)
sim_threshold = SIM_THRESHOLD



# for text in test_set:
new_text = attack.cloak_textfooler(text,classifier_func,importance_func,select_func,synonym_func,candidate_word_filter, candidate_sentence_filter,sim_func,sim_threshold)
