import csv
import os
import time

import numpy as np
from torch.utils.data import Dataset

from GoGame import GoGame


class GoDataset(Dataset):
    def __init__(self, path_of_data, length):
        """
        Initializes the GoDataset with the given CSV file path.
        Args:
            path (str): Path to the CSV file containing Go game data.
        """
        super().__init__()
        self.path = path_of_data
        self.preprocessed_path = "data/preprocessed data"
        self.length = length
        self.goGame = GoGame()
        self.char2idx = {c: i for i, c in enumerate("abcdefghijklmnopqrs")}
        self.dir_len = len(os.listdir('data/preprocessed data'))

        # Load data from CSV file
        with open(self.path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            # Read row by row
            self.data = list(reader)  # dtype: list[str]

    def __rotate_board(self, board, n):
        board = torch.rot90(board, k=n, dims=(2, 3))

        return board

    def __read_from_file(self, row):
        # get filename
        filename = os.path.join(self.preprocessed_path, f'subdir_{int(row[0][2:])%self.dir_len}', row[0])
        boards = torch.load(filename + ".pt").to(
            dtype=torch.float32
        )

        random_start = np.random.randint(0, len(boards) - self.length)
        boards = boards[random_start : random_start + self.length]

        # get label
        self.goGame.reset()
        dim = 0 if row[random_start + self.length][0] == "B" else 1
        self.goGame.place_stone(
            self.char2idx[row[random_start + self.length][2]],
            self.char2idx[row[random_start + self.length][3]],
            dim,
        )
        label = self.goGame.get_board().clone()

        # add a board to the end of the sequence
        # color_board = torch.zeros((2, 19, 19), dtype=torch.float32)
        # color_board[dim] = torch.ones((19, 19), dtype=torch.float32)
        # boards = torch.cat((boards, color_board.unsqueeze(0)), dim=0)

        # rotate boards and label
        # boards = self.__rotate_board(boards, self.rotate_times)
        # label = label.rot90(self.rotate_times, dims=(1, 2))
        label = label.reshape(-1)

        return boards, label

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

        # Randomly rotate times
        self.rotate_times = np.random.randint(3)

        # Transform data into a board
        processed_data, label = self.__read_from_file(row)
        return processed_data, label
