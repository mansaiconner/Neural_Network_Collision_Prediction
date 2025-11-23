import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        # balance collision vs non-collision samples
        zeros = np.where(self.data[:, -1] == 0)[0]
        ones = np.where(self.data[:, -1] == 1)[0]
        #print(ones)
        np.random.shuffle(zeros)
        balanced_zeros = zeros[:len(ones)]
        balanced_indices = np.concatenate((balanced_zeros, ones))
        np.random.shuffle(balanced_indices)
        self.data = self.data[balanced_indices] 
        #print(self.data)
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms only the sensor and action data
        #self.normalized_data = np.concatenate( (self.normalized_data, self.data[:,6:].reshape(-1,1)) , axis=1) #append collision data without normalization 
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
        #print(self.normalized_data, self.normalized_data.shape)
        #print("Data", self.data[:5,:])
        #print("Normalized Data", self.normalized_data[:5,:])
        
    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        item = { 'input' : self.normalized_data[idx, 0:6].astype(np.float32), 'label' : self.normalized_data[idx, 6].astype(np.float32) }
        return item


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        
        # Gets the size of the dataset
        dataset_size = len(self.nav_dataset)
        # Calculates the size of the training set (80% of the dataset)
        train_size = int(0.8 * dataset_size)
        # The remaining samples will be used for the test set
        test_size = dataset_size - train_size

        # split collision and non-collision samples
        #collision_indices = []
        #for i in range(dataset_size):
        #    if self.nav_dataset[i]['label'] == 1.0:
        #        collision_indices.append(i)
        
        #non_collision_indices = []
        #for i in range(dataset_size):
        #    if self.nav_dataset[i]['label'] == 0.0:
        #        non_collision_indices.append(i)
        
        #np.random.shuffle(collision_indices)
        #np.random.shuffle(non_collision_indices)    
        # Create balanced train and test indices
        #train_collision_size = int(0.8 * len(collision_indices))
        #train_non_collision_size = int(0.8 * len(non_collision_indices))
        #train_indices = collision_indices[:train_collision_size] + non_collision_indices[:train_non_collision_size]
        #test_indices = collision_indices[train_collision_size:] + non_collision_indices[train_non_collision_size:]
        #np.random.shuffle(train_indices)
        #np.random.shuffle(test_indices) 

        

        # Create data samplers for training and testing
        train_sampler = data.SubsetRandomSampler(range(0, train_size))
        test_sampler = data.SubsetRandomSampler(range(train_size, dataset_size))

        # Create data loaders for training and testing
        self.train_loader = data.DataLoader(self.nav_dataset, batch_size=batch_size, sampler=train_sampler)
        self.test_loader = data.DataLoader(self.nav_dataset, batch_size=batch_size, sampler=test_sampler)
        #print(self.test_loader)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
