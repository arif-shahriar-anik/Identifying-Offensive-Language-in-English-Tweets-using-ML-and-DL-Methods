### TODO:

#### Dependencies

List all libray and/or other dependencies needed to run your code. Start by stating in detail the environment wher you could run your code? Was it a lab machine? if so, which one? Was it your own computer? If so, what is the setup?

If you can, provide a Docker file to let the TAs run your code.

We ran the code on our personal computers. Codes ran successfully on both Windows and Mac operating system. We downloaded all the necessary libraries before running the code. We used jupyter notebook, Spyder IDE, annaconda prompt and even the terminal for running our codes.

The pre-defined modules and libraries used are:
1) numpy
2) pandas
3) wget
4) os
5) sklearn
6) nltk
7) re
8) collections
10) keras
11) tokenization (python file already provided with codes)
12) tensorflow
13) tensorflow_hub
14) sentencepiece

#### Instructions

Add step-by-step instructions so that the TA can run your code.

For running 501Project_FinalTASK1.py, 501Project_FinalTASK2 required libraries should be downloaded. The datasets for Task A ("Final Data Version 3.csv", "Test Data.csv") and Task B ("Final Data Task B.tsv", "Test Data Task B.tsv") should be kept in same folder to successfully execute the codes. The random seed value was set so that it produce similar results in all runs.

For the first run, nltk.download('averaged_perceptron_tagger') command should be used to download the tagger which facilitates pos tag and lemmatization task. All the preprocessings steps are already included in the code for train and test data. After first run, it is possible to try the code for TaskA after dropping few tweets value (dropping code is commented out) to experiment with results when train and test data have similar class distribution.

For BERT Task 1, model training was performed for 1 epoch with batch-size of 32. It might require tokenization.py module which is already provided with codes in Github folder. Model training might require time and the final macro f1-score is reported in last line of code.
