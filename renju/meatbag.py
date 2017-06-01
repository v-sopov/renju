import numpy as np

class Meatbag:
    def __init__(self):
        self.board = np.zeros((15, 15))
    def make_move(self, *args):
        while True:
            # many more possible problems
            raw = input()
            width = ord(raw[0].lower())
            if width > ord('i'):
                width -= 1
            width -= ord('a')
            height = int(raw[1:]) - 1
            if round(self.board[height, width]) == 0:
                break
            print('Wrong action')
        self.board[height, width] = -1.0 - 2*self.board.sum()
        return (height, width)

    def pass_move(self, action):
        self.board[action] = -1.0 - 2*self.board.sum()
