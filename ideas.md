This file is for scribbling down any ideas we have, to be refined or used as inspiration later.

# Letter transposition experiment
* Explore sentiment analysis for classifier?
* Search for flame wars on Reddit to use as input data
* Examine science of typos so as to avoid existing typos when creating attacks; perhaps look for something like a blossom matrix?

# General adversarial ideas
* Look at meta behavior of hybrid models, look for things that are usually correlated and flag examples that aren't as possible attacks
* A lot of these attack patterns are incredibly obvious to humans; can we train something to recognize it's under attack?
* Visual element is generally ignored by text analysis tools; whitespace is removed in preprocessing, everything is lowercased, that sort of thing. Should be easy to make certain words jump out at humans while machine doesn't notice anything special; for example, in THIS sentence only a WORD or two are IMPORTANT, the rest is just filler that can be used to keep key words separated to machine eyes (for example, word embedding training windows are generally only 5-10 words across).
