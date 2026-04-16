import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


SEEDS = [14, 51, 60, 71, 92]

TEST_HDF5_PATH = 'DB/test/gray_test.h5'
MODEL_BASE_DIR = 'experiments/XIU_20/batch128_lr1e-05'

BATCH_SIZE = 2048
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# 1. Model Architecture (Same as Training — unchanged)
# ---------------------------------------------------------

class xiu_20(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 3), stride=(3, 1), dilation=(2, 1), padding=(3, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(3, 1)),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 3), padding=(2, 1)),
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
        x = x.reshape(-1, 1, 64, 60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 46080)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


# ---------------------------------------------------------
# 2. Test Dataset
# ---------------------------------------------------------

class TestHDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5_file = None
        with h5py.File(self.hdf5_path, 'r') as f:
            self._len = len(f['images'])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

        image_np = self.h5_file['images'][idx]
        date = self.h5_file['dates'][idx]
        permno = self.h5_file['permnos'][idx]

        image_tensor = torch.from_numpy(image_np)
        return image_tensor, int(date), int(permno)


# ---------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------

def get_predictions(seed, model_dir, dataloader, device):
    model = xiu_20()

    model_path = os.path.join(model_dir, f'seed{seed}', 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    probs = []

    print(f">> Inference for Seed {seed}...")
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, desc=f"Seed {seed}", leave=False):
            images = images.to(device, non_blocking=True)
            images = images.to(torch.float32) / 255.0

            outputs = model(images)

            batch_probs = torch.softmax(outputs, dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())

    return probs


# ---------------------------------------------------------
# 4. Main
# ---------------------------------------------------------

if __name__ == "__main__":

    os.makedirs('result', exist_ok=True)

    test_dataset = TestHDF5Dataset(TEST_HDF5_PATH)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=16, pin_memory=True
    )

    print(f"Test samples: {len(test_dataset):,}")

    # Load dates & permnos
    with h5py.File(TEST_HDF5_PATH, 'r') as f:
        all_dates = f['dates'][:]
        all_permnos = f['permnos'][:]

    df = pd.DataFrame({
        'date': all_dates,
        'permno': all_permnos
    })

    for seed in SEEDS:
        try:
            seed_probs = get_predictions(seed, MODEL_BASE_DIR, test_loader, DEVICE)
            df[f'prob_{seed}'] = seed_probs
            print(f"  Seed {seed}: done")
        except Exception as e:
            print(f"  Error processing seed {seed}: {e}")

    df.to_csv("result/gray_prediction.csv", index=False)
    print("\nDone!")