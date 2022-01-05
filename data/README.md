### TODO

Add a brief description of the dataset here, explain which file(s) have the training data, and which have the test data.

Add links to the website where the data can be downloaded from.

------------------------------------------------------------------

The dataset that we are using is the **SOLID Dataset**. 
**Reference**: Rosenthal, S., Atanasova, P., Karadzhov, G., Zampieri, M., & Nakov, P. (2021, August). SOLID: A Large-Scale Semi-Supervised Dataset for Offensive Language Identification. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021 (pp. 915-928).
https://zenodo.org/record/3950379#.YbBDdS9Q2Ru

▫️It consists of 9 million English tweets for the training set and around 4 thousand tweets for the test set.\
▫️The file given for the **training data** is a tsv file with 3 columns: (1) id (2) average (3) std\
Here:\
-ID refers to the tweet IDs\
-Average is the average of the confidences predicted by several supervised models for a specific instance to belong to the positive class for that subtask\
-Std is the confidences’ standard deviation from average confidences for a particular instance.\
▫️The **test data** for each task is in 2 tsv files.\
-For task 1, one file contains the IDs and tweets and the other the corresponding ID and label of whether the tweet is offensive or not in the form OFF or NOT. \
-Task 2 contains a tsv file with IDs and tweets and the other whether a tweet ID is TIN- targeted or UNT- untargeted.

We have scraped, cleaned and pre-processed the data and separated them into **4 files**- \
▫️'Final Data Version 3.csv' in the code uploaded corresponds to the Train Data for Task A.\
-This file contains 4 columns: (1) id (2) average (3) std (4) tweet [9,768 data]\
▫️'Test Data.csv' in the code uploaded corresponds to the Test Data for Task A.\
-This file contains 4 columns: (1) id (2) labels (3) tweet (4) target [3,887 data]\
▫️'Final Data Task B.tsv' in the code uploaded corresponds to the Train Data for Task B.\
-This file contains 4 columns: (1) id (2) average (3) std (4) tweet [9,908 data]\
▫️'Test Data Task B.csv' in the code uploaded corresponds to the Test Data for Task B.\
-This file contains 4 columns: (1) id (2) labels (3) tweet (4) target [1,422 data]

The 4 files have been uploaded here for ease of use.

[Test Data.csv](https://github.com/UOFA-INTRO-NLP-F21/f2021-proj-arif-shahriar-anik/files/7673621/Test.Data.csv)
[Final Data Version 3.csv](https://github.com/UOFA-INTRO-NLP-F21/f2021-proj-arif-shahriar-anik/files/7673619/Final.Data.Version.3.csv)
[Final Data Task B.csv](https://github.com/UOFA-INTRO-NLP-F21/f2021-proj-arif-shahriar-anik/files/7673624/Final.Data.Task.B.csv)
[Test Data Task B.csv](https://github.com/UOFA-INTRO-NLP-F21/f2021-proj-arif-shahriar-anik/files/7673625/Test.Data.Task.B.csv)

**Note:** The .tsv files had to be changed to .csv as they were not being supported by GitHub and could not have been otherwise uploaded here in the README.md File.



