import torch


class GoGame:
    """
    Go game class.
    This class implements the Go game logic to be used for training the neural network.
    """

    def __init__(self, board_size=19) -> None:
        """
        Initializes the Go game with the given board size.
        Args:
            board_size (int): Size of the Go board (default is 19).
        """

        self.board_size = board_size
        self.board = torch.zeros((board_size, board_size, 2), dtype=torch.float32)

    def place_stone(self, x, y, dim) -> None:
        """
        Places a stone of the specified color at the given position (x, y) on the board.
        Args:
            x (float): X-coordinate of the position.
            y (float): Y-coordinate of the position.
            dim (float): Color of the stone (0 for black, 1 for white).
        """
        self.board[x][y][dim] = 1

    def get_board(self) -> torch.Tensor:
        """
        Returns the current game board.
        Returns:
            torch.Tensor: Current game board.
        """
        return self.board

    def reset(self) -> None:
        """
        Resets the game board to the initial state.
        """
        self.board = torch.zeros(
            (self.board_size, self.board_size, 2), dtype=torch.float32
        )
