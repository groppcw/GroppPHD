# Software components

##IMPORTANT NOTE:
Code development moved to HideAndSeek repo for the time being. This checklist may be out of date.

* DONE - Cross-validation data chopper.
	* Intake; training data.
	* Intake; 10-fold vs 5-fold etc
	* Inputs; training data, output prefix, crossfold
	* Outputs; training and test file pairs
* Wrapper for PLDA.
	* Intake; training data.
	* Intake; PLDA installation location.
	* Output; topic model files (beta matrix)
* EXTERNAL: PLDA
* Wrapper for topic inference.
	* Intake; topic model (beta matrix)
	* Intake; document(s)
	* Output; topic mixture for document(s)
* EXTERNAL: topic inference
* Wrapper for classifier training.
	* Intake; labeled documents
	* Intake; topic mixtures for those documents
	* Output; classifier model file
* Training code for classifier training.
* Execution wrapper for classifier.
	* Intake; classifier model file
	* Intake; test document(s)
	* Output; classification of document(s)
* Test code for classifier.
* Attack creator.
	* Intake; training data.
	* Intake; test results.
	* Output; attack code.
* Wrapper to run generated attack.
	* Intake; attack code.
	* Output; attack posts.
* Test code for attack posts.
	* Intake; classifier.
	* Intake; attack posts.
	* Output; scoring of attack posts.
* Spellcheck countermeasure preprocessing.
* Wrapper to rerun testing with the countermeasure.
	* Intake; countermeasure.
	* Intake; classifer.
	* Intake; test data.
	* Intake; attack data.
	* Output; new scores.
