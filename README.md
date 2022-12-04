# CSE6250-Att-BiLSTM

Implementation of [Assertion Detection in Clinical Natural Language Processing: A Knowledge-Poor Machine Learning Approach](https://ieeexplore.ieee.org/document/8710921).

## Dependencies
* python 3.6
* pytorch 1.3.0
* nltk 3.4.5
* numpy 1.19.2
* matplotlib 3.1.0
* scikit-learn 1.1.3

## Data
The corpus for this task and the annotations are available for download at the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/) and [Google Code Archive](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/negex/rsAnnotations-1-120-random.txt).
* i2b2-BID/PH
* i2b2-UPMC
* NegEx-Corp

## Preprocessing 
1. Download raw data and decompress it into `data` folder.
2. Run the following commands to convert the raw data to the specified format.
```shell
python preprocess.py
```
3. The conversion results are `train.json` and `test.json`.
4. In addition, create a file `relation2id.txt`.

## Word Embedding
1. Download the following 2 files to this 'embedding' folder:
    * [glove.6B.200d.txt.tgz](http://choicenetworks.com/Downloads/glove.6B.200d.txt.tgz)
    * [SemEval2010_task8_all_data.tgz](http://choicenetworks.com/Downloads/SemEval2010_task8_all_data.tgz) 

2. Decompress the 2 files, e.g.:
```shell
tar -zxvf glove.6B.200d.txt.tgz
tar -zxvf SemEval2010_task8_all_data.tgz
```

## Instruction to run the code
1. Run the following commands to start the program.
```shell
python run.py
```
2. Run the follwing commands to see more details.
```shell
python run.py -h
```


