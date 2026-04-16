import os
import gc
import h5py
import random
import numpy as np

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset

import wandb
import argparse

def save_checkpoint(epoch, model, optimizer, loss, filename='checkpoint.pth.tar'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5_file = None
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self._len = len(f['labels'])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        image_np = self.h5_file['images'][idx] # Shape: (1, 64, 60), dtype: uint8
        label_raw = self.h5_file['labels'][idx]
        
        label = 1 if label_raw > 0 else 0
        
        image_tensor = torch.from_numpy(image_np)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)

class xiu_20(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1),padding=(3, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3),padding=(3, 1)),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3),padding=(2, 1)),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(46080, 2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1,1,64,60)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.reshape(-1,46080)
        x = self.fc1(x)
        x = self.softmax(x)

        return x

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def seed_worker(worker_id, main_seed):
    worker_seed = main_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
def main(seed):

    batch_size = 128
    learning_rate = 0.00001 # 10^-5

    set_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False

    HDF5_DATASET_PATH = 'DB/train/gray_train.h5'
    INDICES_PATH = f'DB/train/split_indices/gray/seed_{seed}.npz'

    indices_data = np.load(INDICES_PATH)
    train_indices = indices_data['train_indices']
    valid_indices = indices_data['valid_indices']

    full_dataset = HDF5Dataset(hdf5_path=HDF5_DATASET_PATH)

    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, generator=g, worker_init_fn = lambda worker_id: seed_worker(worker_id, seed), pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, generator=g, worker_init_fn = lambda worker_id: seed_worker(worker_id, seed), pin_memory=True)

    result_path = f'experiments/XIU_20/batch{batch_size}_lr{learning_rate}/seed{seed}'
    os.makedirs(result_path + '/',exist_ok=True)

    # Define the model and move it to the device
    model = xiu_20()
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)
    else:
        print("GPU is not available.")
        exit()

    # Initialize wandb
    wandb.init(project='chart-vit-xai', config={
        'color' : 'GRAY',
        'seed': seed,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'memo' : 'XIU_20'
    })

    # Training settings
    num_epochs = 100
    patience = 2  # early stopping patience
    best_val_loss = float('inf')

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            images = images.to(torch.float32) / 255.0

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / total
        train_loss /= total

        # Validation loop
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)

                images = images.to(torch.float32) / 255.0

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        valid_acc = correct / total
        valid_loss /= total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Train Accuracy: {train_acc:.4f}, Valid Loss: {valid_loss:.8f}, Valid Accuracy: {valid_acc:.4f}')
    
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'valid_loss': valid_loss,
            'valid_accuracy': valid_acc
        })

        np.savetxt(result_path + '/train_losses.txt', train_losses)
        np.savetxt(result_path + '/valid_losses.txt', valid_losses)
        np.savetxt(result_path + '/train_accs.txt', train_accs)
        np.savetxt(result_path + '/valid_accs.txt', valid_accs)

        save_checkpoint(epoch, model, optimizer, valid_loss, os.path.join(result_path, 'checkpoint.pth.tar'))
        
        # Check for early stopping
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(result_path, 'best_model.pth'))
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print('Early stopping triggered')
            break
        
        # Free up memory
        gc.collect()
        torch.cuda.empty_cache()
        
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.seed)