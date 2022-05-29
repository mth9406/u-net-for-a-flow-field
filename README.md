# Fluid U net 
A PyTorch implementation of the MCDAE

Data in this repo is fluid field around a cylinder. 
The copyright of the data is at https://fpe.postech.ac.kr/postech/.
If you have access to it, you can (create and) put it in the folder `data/`.

# How to use
First of all, download the data from the data source above.    
and then, run main.py. Below describes how to train a model using main.py.

```bash
usage: main.py [-h] [--dataset_path DATASET_PATH] [--test_size TEST_SIZE]
               [--val_size VAL_SIZE] [--batch_size BATCH_SIZE] [--epoch EPOCH]
               [--lr LR] [--patience PATIENCE] [--delta DELTA]
               [--model_path MODEL_PATH] [--print_log_option PRINT_LOG_OPTION]
               [--model_type MODEL_TYPE] [--latent_dim LATENT_DIM]
               [--lags LAGS] [--test] [--model_file MODEL_FILE]
```

# Optional arguments
```
optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
  --test_size TEST_SIZE
                        test size when splitting the data (float type)
  --val_size VAL_SIZE   validation size when splitting the data (float type)
  --batch_size BATCH_SIZE
                        input batch size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --patience PATIENCE   patience of early stopping condition
  --delta DELTA         significant improvement to update a model
  --model_path MODEL_PATH
                        a path to sava a model if test is false, otherwise it
                        is a path to a model to test
  --print_log_option PRINT_LOG_OPTION
                        print training loss every print_log_option
  --model_type MODEL_TYPE
                        model type: 0 for a heavy model 1 for a light model
  --latent_dim LATENT_DIM
                        latent dimension
  --lags LAGS
  --test                test
  --model_file MODEL_FILE
                        model file
```

Please run the "example_usage.ipynb" if you find anything hard to understannd. 