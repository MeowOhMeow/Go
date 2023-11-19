import csv
import os

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from GoGame import GoGame


class GoDataset(Dataset):
    def __init__(self, path_of_data, data_len):
        """
        Initializes the GoDataset with the given CSV file path.
        Args:
            path (str): Path to the CSV file containing Go game data.
        """
        super().__init__()
        self.path = path_of_data
        self.preprocessed_path = "data/preprocessed data"
        self.char2idx = {c: i for i, c in enumerate("abcdefghijklmnopqrs")}
        self.dir_len = len(os.listdir('data/preprocessed data'))
        self.data_len = data_len

        # Load data from CSV file
        with open(self.path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            # Read row by row
            self.data = list(reader)  # dtype: list[str]
    

    def _read_from_file(self, row):
        # get filename
        filename = os.path.join(self.preprocessed_path, f'subdir_{int(row[0][2:])%self.dir_len}', row[0])
        
        # get boards
        boards = torch.load(filename + ".pt").to(
            dtype=torch.float32
        )
        # discard the last board, this board doesn't have a label
        boards = boards[:-1]
        
        max_len = len(boards)

        random_start = np.random.randint(0, max_len - self.data_len)
        boards = boards[random_start:random_start + self.data_len]

        color = 1 if row[random_start + 2][0] == 'B' else 0
        colors = torch.empty((self.data_len, 1), dtype=torch.float32)
        for i in range(self.data_len):
            if i % 2 == color:
                colors[i] = 1
            else:
                colors[i] = -1
        # get label
        x = self.char2idx[row[random_start + self.data_len + 3][2]]
        y = self.char2idx[row[random_start + self.data_len + 3][3]]
        label = torch.zeros((361), dtype=torch.float32)
        label[x * 19 + y] = 1

        return boards, label, colors
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get data at the given index.
        Args:
            idx (int): Index of the data sample.
        Returns:
            torch.Tensor: Processed and padded data sample.
        """
        # Get data at the given index
        row = self.data[idx]

        return self._read_from_file(row)

