from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        
        for epoch in range(num_epochs):  
            ############
            # Training #
            ############
            
            running_tr_loss = 0.0
            counter = 0
            for batch_idx, (X_tr_batch, Y_tr_batch) in enumerate(train_loader):
                counter+=1
                if torch.cuda.is_available():
                    X_tr_batch, Y_tr_batch = X_tr_batch.cuda(), Y_tr_batch.cuda()
                
                ## zero the parameter gradients
                optim.zero_grad()
                
                output_tr = model(X_tr_batch)
                train_loss = self.loss_func(output_tr, Y_tr_batch)
                
                # computes dloss/dw for every parameter w which has requires_grad=True
                train_loss.backward()
                
                # updates the parameters
                optim.step()
                
                # print statistics
                if log_nth == counter:  
                    print("[Iteration %d/%d] TRAIN loss: %.3f" % (batch_idx + 1, iter_per_epoch, train_loss.item()))
                    counter = 0
                    
                running_tr_loss += train_loss.item()
                    
            # save loss for plotting
            self.train_loss_history.append(running_tr_loss/len(train_loader))

            # save accuracy for plotting
            pred = torch.argmax(output_tr, dim=1)
            train_acc = torch.sum(pred == Y_tr_batch).item()/train_loader.batch_size
            self.train_acc_history.append(train_acc)
            
            ##############
            # Validation #
            ##############
            running_val_loss = 0.0
            for batch_idx, (X_val_batch, Y_val_batch) in enumerate(val_loader):
                output_val = model(X_val_batch)
                val_loss = self.loss_func(output_val, Y_val_batch)
                running_val_loss += val_loss.item()
            
            # save loss for plotting
            self.val_loss_history.append(running_val_loss/len(val_loader))
            
            # save accuracy for plotting
            pred = torch.argmax(output_val, dim=1)
            val_acc = torch.sum(pred == Y_val_batch).item()/val_loader.batch_size
            self.val_acc_history.append(val_acc)
            
            
            print("[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f" % (epoch + 1, num_epochs, train_acc,
                                                               running_tr_loss/len(train_loader)))
            print("[Epoch %d/%d] VAL acc/loss: %.3f/%.3f" % (epoch + 1, num_epochs, val_acc, 
                                                             running_val_loss/len(val_loader)))
                
                
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')