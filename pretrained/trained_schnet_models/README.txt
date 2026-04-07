How to use the trained SchNet models:

- Every directory contains the best performing model ('best_model') based on the
validation set, the arguments that have been used for training ('args.json') and the
train-val-test-split ('split.npz').

- The trained models can be used with:
"spk_run.py eval <path-to-database> <path-to-modeldir> --split test". This will evaluate
the model on the test set and save the mean absolute error to 'evaluation.csv'.

- In order to use trained models for your own experiments use:
model = torch.load("<modeldir>/best_model", map_location=<device>)

- The split file can be used to reproduce the train-val-test-split of the dataset:
train, val, test = spk.data.train_test_split(dataset, split_file=<split_path>)
