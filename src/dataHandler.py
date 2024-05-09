import os
import random
import torch
import wfdb
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import biosppy.signals.ecg as bp
from wfdb import processing
from typing import List,Tuple, Sequence
from torch.utils.data import Dataset
from torchvision import transforms


class ECGDataset(Dataset):
    def __init__(self, data, labels, transform=None, extra=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.extra = extra

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        indices = np.random.permutation(len(self.data))
        data_sample = self.data[indices[idx]]
        label = self.labels[indices[idx]]
        
        if self.transform:
            data_sample = self.transform(data_sample)
        
        return data_sample, label

    def return_knowledge_dict(self):
        return self.extra
    
    def split_dataset(self, train_ratio=0.8, shuffle=True):
        # Identify unique classes
        unique_classes = np.unique(self.labels)

        # Initialize lists to store indices for each class
        train_indices = []
        val_indices = []

        # Split indices for each class into train and validation sets
        for cls in unique_classes:
            cls_indices = np.where(self.labels == cls)[0]
            np.random.shuffle(cls_indices)  # Shuffle indices

            # Split indices based on train_ratio
            split_idx = int(train_ratio * len(cls_indices))
            train_idx = cls_indices[:split_idx]
            val_idx = cls_indices[split_idx:]

            train_indices.extend(train_idx)
            val_indices.extend(val_idx)

        # Shuffle indices if required
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

        return train_indices, val_indices


class ECG_DATAHANDLER:
    def __init__(self, users_info: dict) -> None:
        self.users_info: dict = users_info
        self.window_size: int = 300

    @classmethod
    def gather_records_info(cls,path: str) -> dict:

        users_info = {}
        total_person_directories = 0
        users = []
        # Iterate over directories
        for root, dirs, _  in os.walk(path):
            # Count directories starting with 'Person_'
            for directory in dirs:
                if directory.startswith("Person_"):
                    user = {}
                    total_person_directories += 1
                    # Count files within each 'Person_' directory
                    person_dir_path = os.path.join(root, directory)
                    records = fnmatch.filter(os.listdir(person_dir_path), "rec_*.dat")
                    data_paths = [os.path.join(person_dir_path, os.path.splitext(file)[0]) for file in records]
                    if (len(data_paths) > 1):
                        user["name"]= directory
                        user["num_records"] = len(data_paths)
                        user["data_paths"]= data_paths
                        users.append(user)

        users_info['num_users'] = len(users)
        users_info['users'] = users

        return cls(users_info)
        
    def prepare_data(self) -> None:
        beats_dict: dict = {}

        for user in self.users_info['users']:
            #Sample 2 random records
            print("Processing user "+user['name'])
            sampled_records = random.sample(range(0, user['num_records']), 2)
            beats = []
            for i in range(len(sampled_records)):
                #Gather the filtered signal
                pth: str = user['data_paths'][sampled_records[i]]
                sig, fields = wfdb.rdsamp(pth, channel_names=['ECG I filtered'])
                _, _, rpeaks,_, heartbeats,_,_ = bp.ecg(signal=sig[:,0], sampling_rate=fields['fs'], path=None, show=False, interactive=False)
                beats.extend(heartbeats)

            if len(beats)>40: #Remove low data users
                print(len(beats), "beats found")
                beats_dict[user['name']]=beats
                for b in beats:
                    plt.plot(b)
                plt.title('Superposed ECG_beats-'+user['name'])
                plt.grid()
                plt.savefig("./ECG_ID_dataset/plots/beats/"+user['name'])
                plt.close('all')

        self.beats_dict = beats_dict

    def _create_the_user_labels(self):
        knowledge_dict = {}
        training_labels =  []
        for index, (k,v) in enumerate(self.beats_dict.items()):
            knowledge_dict[k]=index
            for _ in range(len(v)):
                training_labels.append(index)
        
        return training_labels, knowledge_dict
    
    class Normalize1D(object):
        def __call__(self, data):
            min_val = torch.min(data)
            max_val = torch.max(data)
            normalized_data = (data-min_val)/(max_val-min_val)
            return normalized_data
    

    def transform_data_to_dataset(self, path: str, action: str="SAVE"):
        if (action == "SAVE"):
            self.prepare_data()
            user_labels, knowledge_dict= self._create_the_user_labels()
            sub_tensors=[]

            for username in self.beats_dict:
                st = [torch.tensor(i, dtype=torch.float32) for i in self.beats_dict[username]]
                sub_tensors += st

            transform = transforms.Compose([
                self.Normalize1D()  # Normalize data
            ])

            tensor_data= sub_tensors
            dataset = ECGDataset(tensor_data, user_labels, transform=transform, extra = knowledge_dict)

            torch.save(dataset, path)  
            print("Data saved for %d users and ready to be used in"%(len(self.beats_dict.keys())) + path)
            return dataset, len(self.beats_dict.keys())
        else:
            if os.path.exists(path):
                dataset = torch.load(path)
                return dataset, len(set(dataset.labels))
            else:
                print("File %s does not exist. Creating a new dataset"%(path))
                return self.transform_data_to_dataset(path,action="SAVE")
            
    def transform_df_to_dataset(self, df):
        X = df.iloc[:, :-1]  # Features (all columns except the last one)
        y = df.iloc[:, -1]   # Target variable (last column)
        
        tensor_X = torch.tensor(X.values,dtype=torch.float32)
        tensor_y = torch.tensor(y.values, dtype=torch.int)

        dataset = ECGDataset(tensor_X, tensor_y)

        return dataset
