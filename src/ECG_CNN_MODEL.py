import torch
import time
import numpy as np
import seaborn as sns
import copy
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix,precision_recall_fscore_support

class MetricsHolder:
    def __init__(self, y_true, y_pred, scores) :
        self.y_true=y_true
        self.y_pred=y_pred
        self.scores = scores
        self.accuracy=accuracy_score(y_true,y_pred)

    def calculate_recognition_rate(self):
        return round(self.accuracy*100, 4)
    
    def rank_n_accuracy(self, n):
        correct = 0
        for i, true_label in enumerate(self.y_true):
            top_n_indices = np.argsort(self.scores[i])[-n:]  # Indices of the top N scores
            if true_label in top_n_indices:
                correct += 1
        return round(correct *100 / len(self.y_true), 4)
    
    def calculate_weighted_precision(self):
        return round(precision_score(self.y_true, self.y_pred, zero_division=1, average='weighted')*100,4)
    
    def calculate_weighted_recall(self):
        return round(recall_score(self.y_true, self.y_pred, average='weighted')*100,4)
    
    def calculate_weighted_f1_score(self):
        return round(f1_score(self.y_true, self.y_pred, average='weighted')*100,4)
    
    
    def plot_cmc_curve(self, max_rank=5):
        cmc = np.zeros(max_rank)
        for i, true_label in enumerate(self.y_true):
            sorted_scores_indices = np.argsort(self.scores[i])[::-1]  # Indices of scores sorted in descending order
            rank = np.where(sorted_scores_indices == true_label)[0][0] + 1
            if rank <= max_rank:
                cmc[rank-1:] += 1

        cmc = cmc / len(self.y_true)

        plt.plot(range(1, max_rank + 1), cmc, marker='o')
        plt.xlabel('Rank')
        plt.ylabel('Recognition Rate')
        plt.title('CMC Curve')
        plt.grid(True)
        plt.show()
    

class EarlyStopper:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state_dict = None

    def __call__(self, val_loss, curr_model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state_dict = copy.deepcopy(curr_model.state_dict())

        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EARLY STOPPER counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: #Improvement!
            self.best_score = val_loss
            self.counter = 0
            self.best_model_state_dict = copy.deepcopy(curr_model.state_dict())
        return self.early_stop, self.best_model_state_dict #I'll return the model that reached the best validation

class ECG_1D_CNN(nn.Module):
    def __init__(self,beatLength,nLabels):
        super().__init__()

        #250x1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)

        self.pool =  nn.MaxPool1d(kernel_size=4, stride=2, padding=0)

        #Spatial dimension of the signal at the output of the 2nd ConvLayer and 2ndPool!
        self.final_dim = 73
        self.final_channels = 64
        print(f"Nusers:{nLabels}, beat Length: {beatLength} datapoints.")

        #Dense Layers
        self.linear1 = nn.Linear(self.final_dim*self.final_channels,   int(self.final_dim*self.final_channels/2))
        self.linear2 = nn.Linear(int(self.final_dim*self.final_channels/2), int(self.final_dim*self.final_channels/8))
        self.linear3 = nn.Linear(int(self.final_dim*self.final_channels/8), nLabels)

        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1) 

        
    def forward(self, x, reg=True):
        # Pass the input tensor through the CNN operations
        x = self.conv1(x)
        if reg:
            x = self.batch_norm1(x) 
        x = self.relu(x) 
        x = self.pool(x)
        x = self.conv2(x)
        if reg:
            x = self.batch_norm2(x) 
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the tensor into a vector of appropriate dimension using self.final_dim
        x = x.view(-1, self.final_dim*self.final_channels) 

        # Pass the tensor through the Dense Layers
        if reg:
            x = nn.Dropout(p=0.5)(x)
        x = self.linear1(x)
        x = self.relu(x)
        if reg:
            x = nn.Dropout(p=0.5)(x)
        x = self.linear2(x)
        x = self.relu(x)
        if reg:
            x = nn.Dropout(p=0.3)(x)
        x = self.linear3(x)
        x = self.logsoftmax(x) 
        return x


class ECG_1D_CNN_TRAINER(ECG_1D_CNN):
    def __init__(self,beatLength,nLabels,epochs=100,lr=0.001):
        super().__init__(beatLength,nLabels) 

        self.lr = lr #Learning Rate
        self.optim = optim.Adam(self.parameters(), self.lr)        
        self.epochs = epochs
        self.criterion = nn.NLLLoss()         
        
        self.loss_during_training = [] 
        self.valid_loss_during_training = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DETECTED DEVICE %s"%(self.device))
        self.to(self.device)

    def trainloop(self,trainingLoader,validationLoader, earlyStopper= lambda x,y: (False, None), reg=True):

        tot_t = time.time()
        for e in range(int(self.epochs)):
            start_time = time.time()
            running_loss = 0.
            self.train()

            for beats, labels in trainingLoader:              
                self.optim.zero_grad() 
                beats, labels = beats.to(self.device), labels.to(self.device)  
                out = self.forward(beats.view(beats.shape[0], 1, beats.shape[1]),reg) 

                loss = self.criterion(out,labels)
                running_loss += loss.item()
                loss.backward()
                self.optim.step()
                
            self.loss_during_training.append(running_loss/len(trainingLoader))
            
            # Validation Loss for this epoch
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad(): 
                self.eval() # set model to evaluation mode
                running_loss = 0.
    
                for beats,labels in validationLoader:
                    beats, labels = beats.to(self.device), labels.to(self.device)  
                    out = self.forward(beats.view(beats.shape[0],1,beats.shape[1]),reg) 
                    loss = self.criterion(out,labels) #Evaluation!
                    running_loss += loss.item()   
                        
                self.valid_loss_during_training.append(running_loss/len(validationLoader))    

            if(e % 1 == 0): # Every epoch
                print("Epoch %d. Training loss: %f, Validation loss: %f, Time per epoch: %f seconds" 
                      %(e,self.loss_during_training[-1],self.valid_loss_during_training[-1],
                       (time.time() - start_time)))
                
            #Early stopping callback 
            stop, best_model_dict = earlyStopper(self.valid_loss_during_training[-1], self)  
            if stop:
                print("Early stopping Reached ")
                if(best_model_dict is not None):
                    self.load_state_dict(best_model_dict, assign=True) #best scoring weights
                break
        print("Total training time for %d epochs: %f seconds" %(e+1,(time.time() - tot_t)))

    def eval_performance(self,dataloader):
        self.eval()
        y_pred = []
        y_real = []
        scores = []
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():

            for beats,labels in dataloader:
                # Move input and label tensors to the default device
                beats, labels = beats.to(self.device), labels.to(self.device)  
                probs = self.forward(beats.view(beats.shape[0],1,beats.shape[1])) 
                _, top_class = probs.topk(1, dim=1)
                
                scores.append(probs.cpu().numpy())
                y_pred.extend(top_class.cpu().numpy().T[0])
                y_real.extend(labels.cpu().numpy())

            return MetricsHolder(y_true=y_real, y_pred=y_pred, scores = np.vstack(scores))
        
    def activation_maximization(self, target_class, delta= 1, verbose=False):
        self.eval()  
        rand_beat_tensor = torch.zeros(1, 1, 300, dtype=torch.float32, device=self.device).requires_grad_(True) 
        prob_k1 = 1
        prob_k2 = 1
        found_class = 1000
        i=0

        #While the second most probable class is still close...
        while (abs(prob_k1 - prob_k2) < delta) or found_class != target_class: 
            rand_beat_tensor = rand_beat_tensor.to(self.device)
            rand_beat_tensor.retain_grad()
            output = self.forward(rand_beat_tensor)
            
            # Compute loss: negative log-likelihood
            loss = -torch.log_softmax(output, dim=1)[:, target_class].mean()
            
            # Zero gradients, perform backward pass, and update input tensor
            self.zero_grad() #Prob not needed
            rand_beat_tensor.grad = None  # Clear gradients for input tensor
            loss.backward()
            rand_beat_tensor.data -= 0.01 * rand_beat_tensor.grad
            rand_beat_tensor.data = torch.clamp(rand_beat_tensor.data, 0, 1)
            i+=1

            # Print the loss every few iterations
            if (i) % 500 == 0:
                probs, classes = output.topk(2, dim=1)
                prob_k1 = abs(probs.detach().cpu().numpy()[0][0])
                prob_k2 = abs(probs.detach().cpu().numpy()[0][1])
                found_class = abs(classes.detach().cpu().numpy()[0][0])
                if verbose:
                    print("\tITER %d MOST PROBABLE CLASS:"%(i),found_class)

        # Convert the optimized input image back to the numpy array
        print("\tEND ON ITER %d MOST PROBABLE CLASS:"%(i),found_class)
        optim_ecg = rand_beat_tensor.detach().cpu().numpy()
        return optim_ecg[0][0]
    
    def create_templates(self, knowledge_dict, n_pass, savePath=None, plot=False, save=False):

        templates = []
        for k,v in knowledge_dict.items():
            print("Creating Template For user: ",k)
            for i in range(n_pass):
                template=self.activation_maximization(v,verbose=False)  
                print(np.mean(template))
                templates.append(np.append(template, v)) #Append the label also
                if plot:
                    plt.plot(template)
            if plot:
                plt.title('ECG Template '+ k)
                plt.grid()
                plt.savefig("./ECG_ID_dataset/plots/templates/"+k)
                plt.close('all')

        if save: 
            templates_matrix = np.array(templates)
            np.savetxt(savePath, templates_matrix, delimiter=",", fmt="%s")

        self.templates = templates

