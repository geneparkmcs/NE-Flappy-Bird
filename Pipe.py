class Pipe:
    def __init__(self, x, gap_y, gap_height=150):
        self.x = x
        self.gap_y = gap_y
        self.gap_height = gap_height
        self.width = 50
        self.passed = False

    def update(self):
        self.x -= 5