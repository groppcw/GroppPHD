# Experimental Setup

This file is for detailing the setup and environment requirements of each experiment, along with a description of what the experiment is attempting to achieve.

# Letter transposition Experiment
## Overview
The goal of this experiment is to use a simple scenario to create the necessary framework to explore adversarial AI. In it, we will attack a primitive classifier by creating unusual typos of known words that are still readable to humans, so as to avoid detection. We will use a scenario of attempting to identify social media comments that are pro pineapple pizza as our demonstration.
## Phase One: Training
In this phase, we need to train our classifier. This classifier is probably going to be constructed using a topic model, although we may also explore using sentiment analysis to augment it. To train this, we will need a corpus of data.
* Data could be constructed using a generator, but it would be ideal to find an existing data set we can label. Search for Reddit flame wars, maybe?
* We need a reproducible process for generating our classifier. For using LDA, we could use PLDA (*not* PLDA+, which has race conditions) with fixed seed.
* We need to set up a test set, possibly by simply removing some of our training data set, and confirm that our classifier is functional.
## Phase Two: Attack
In this phase, we attempt to mask our message by modifying critical words into ones the classifier cannot recognize and will thus ignore. Since our training data likely includes common typos, we must take care to avoid those. We also need our words to still read correctly to human viewers, by following the rules discovered by researchers studying human reading skills.
* Identify key words to corrupt.
* Identify pieces of the word that can be corrupted without losing comprehension.
* Corrupt the word, avoiding common typos that may already be known to the detection system.
* Test that our corrupted messages evade the detection system.
## Phase Three: Defend
Come up with a way to modify the detector to respond correctly to the attack messages.
* One possibility; spell correction on unknown words that are close to known words.
