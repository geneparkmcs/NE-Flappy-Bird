# pipe.py

import random

class Pipe:
    def __init__(self, x, gap_y=None, gap_height=150):
        self.x = x
        self.gap_y = gap_y if gap_y is not None else random.randint(100, 400)
        self.gap_height = gap_height
        self.width = 50
        self.passed = False
    
    def update(self):
        self.x -= 5
