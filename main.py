import os
import csv
import argparse
import sys
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from utils import *
from models import *

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--dataset_path', type= str, default= './cylinder/Data/Re300')
parser.add_argument('--test_size', type= float, default= 0.3,
                help= 'test size when splitting the data (float type)')
parser.add_argument('--val_size', type= float, default= 0.1, 
                help= 'validation size when splitting the data (float type)')

# Training args
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--model_path', type=str, default='./', 
                    help='a path to sava a model if test is false, otherwise it is a path to a model to test')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')

# Model configs
parser.add_argument('--model_type', type= int, default= 1, help= 'model type: 0 for a heavy model 1 for a light model')
parser.add_argument('--latent_dim', type= int, default= 2, help= 'latent dimension')
parser.add_argument('--lags', type= int, default= 2)

# To test
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--model_file', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model file', required= False)

args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model 
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

def main():
    
    print("Loading data...")
    # load data
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_fluid_data(fluid_path= args.dataset_path, 
                                                                            lag= args.lags, test_size= args.test_size, valid_size= args.val_size)
    train_data = FluidDataset(X_train, Y_train)
    valid_data = FluidDataset(X_valid, Y_valid)
    test_data = FluidDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)  
    print("Loading data done!")

    # model
    if args.model_type == 0:
        model = FluidalUnet(args.lags, args.latent_dim).to(device)
    elif args.model_type == 1:
        model = LightFluidalUnet(args.lags, args.latent_dim).to(device)
    else:
        print("The model is yet to be implemented.")
        sys.exit()
    
    # setting training args...
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)    
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path
    )    
    if args.test: 
        model_file = os.path.join(args.model_path, args.model_file)
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['state_dict'])

    else:
        logs = {
            'tr_loss':[],
            'valid_loss':[]
        }
        num_batches = len(train_loader)
        print('Start training...')
        for epoch in range(args.epoch):
            # to store losses per epoch
            tr_loss, valid_loss = 0, 0
            # a training loop
            for batch_idx, (x, y) in enumerate(train_loader):

                x, y = x.to(device), y.to(device) 

                model.train()
                # feed forward
                with torch.set_grad_enabled(True):
                    y_hat = model(x)
                    loss = mse(y_hat, y)
                
                # backward 
                model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # store the d_tr_loss
                tr_loss += loss.detach().cpu().item()

                if (batch_idx+1) % args.print_log_option == 0:
                    print(f'Epoch [{epoch+1}/{args.epoch}] Batch [{batch_idx+1}/{num_batches}]: \
                        loss = {loss.detach().cpu().item()}')

            # a validation loop 
            for batch_idx, (x, y) in enumerate(valid_loader):
                x, y = x.to(device), y.to(device)
                
                model.eval()
                loss = 0
                with torch.no_grad():
                    y_hat = model(x)
                    loss = mse(y_hat, y)
                valid_loss += loss.detach().cpu().item()
            
            # save current loss values
            tr_loss, valid_loss = tr_loss/len(train_loader), valid_loss/len(valid_loader)
            logs['tr_loss'].append(tr_loss)
            logs['valid_loss'].append(valid_loss)

            print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {tr_loss:.6f}, validation loss= {valid_loss:.6f}')
            early_stopping(valid_loss, model, epoch, optimizer)

            if early_stopping.early_stop:
                break     

        print("Training done! Saving logs...")
        log_path= os.path.join(args.model_path, 'training_logs')
        os.makedirs(log_path, exist_ok= True)
        log_file= os.path.join(log_path, 'training_logs.csv')
        with open(log_file, 'w', newline= '') as f:
            wr = csv.writer(f)
            n = len(logs['tr_loss'])
            rows = np.array(list(logs.values())).T
            wr.writerow(list(logs.keys()))
            for i in range(1, n):
                wr.writerow(rows[i, :])

    print("Testing the model...")
    te_loss = 0
    te_r2 = 0
    te_mae = 0
    # te_mse = 0
    
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        model.eval()
        with torch.no_grad():
            y_hat_numpy = model(x).detach().cpu().numpy().flatten()
            y_numpy = y.detach().cpu().numpy().flatten()
            te_loss += mean_squared_error(y_hat_numpy, y_numpy)
            te_r2 += r2_score(y_hat_numpy, y_numpy)
            te_mae += mean_absolute_error(y_hat_numpy, y_numpy)
            # te_mse += mean_squared_error(y_hat, y)
    te_loss = te_loss/len(test_loader)
    te_r2 = te_r2/len(test_loader)
    te_mae = te_mae/len(test_loader)
    # te_mse = te_mse/len(test_loader)
    print("Test done!")
    print(f"mse: {te_loss:.2f}")
    print(f"mae: {te_mae:.2f}")
    print(f"r2: {te_r2:.2f}")
    print()
    return

if __name__ == '__main__':
    main()