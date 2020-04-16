This is an application of the ensemble self-training framework on the sentence classification task with the Scicite data set.

Required libraries:
    - Transformers (https://github.com/huggingface/transformers) for the SciBERT encoder
    - Dynet (https://dynet.readthedocs.io/en/latest/) for the classifier model, used in classifier.py, but can be replaced with and ML framework (and code) of your choice, it's independent from the framework


Summary of the files:
    - framework.py defines the general framework and common functionalities, can be used for other tasks
    - utils.py are mostly helper funcitons for the framework
    - scicite_task.py gives the specification of the scicite task, and is the main file to run the experiment
    - classifier.py is the sentence classifier model with a simple Multi-layer Perceptron (MLP)
    - train_scibert.sh and pred_scibert.sh are shellscripts to run the training and prediction of classifier.py; 
    the framework (framework.py) runs these scripts in parallel (since each training procedure runs on one single CPU core);
    by adding more tools and creating train_[TOOL].sh and pred_[TOOL].sh with similar arguments, the framework can use different types of models together, which further increases the diversity of the ensemble
    - make_dataset.py converts the format of the Scicite data into one file that contains all sentences, and another file that contains all the labels
    - encode_text.py encodes all the sentences with SciBERT and store them in a numpy file, to avoid repeated calculation and keep the actual trained model size very small (only an MLP without the SciBERT part)
    - scibert-100.json is the configutation file, which specifies the paths and hyperparameters of the scicite experiment, we use the first 100 sentences from the real training split as the only availble labeled data to simulate low-resource setting


Run the experiment:
    - download the scicite dataset (https://github.com/allenai/scicite), and store in folder ``data''
    - download SciBERT model (https://github.com/allenai/scibert), and store in folder ``models'' (used by encode_text.py)
    - extract the sentences: 
        python make_dataset.py data/train.jsonl data/dev.jsonl data/test.jsonl data/all.txt data/all.tgt
    - encode the sentences: 
        python encode_text.py data/all.txt data/all.scibert.npz
    - run the experiment: 
        python python scicite_task.py exp-1 scibert-100.json 