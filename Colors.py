class Color:
    def __init__(self, b, g, r):
        self.b, self.g, self.r = b, g, r
        if b > r and b > g:
            self.name = "Blue"
        elif g > r and g > b:
            self.name = "Green"
        elif r > b and r > g:
            self.name = "Red"
        elif r > 150 and g > 150:
            self.name = "Yellow"
        else:
            self.name = "Unknown"

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{Color.__name__}({self.b}, {self.g}, {self.r})"
