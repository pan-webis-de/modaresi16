class ZeroOne:
    def compare(self, actual, expected):
        p, r, f1 = 0.0, 0.0, 0.0
        if actual == expected:
            p, r, f1 = 1.0, 1.0, 1.0
        return p, r, f1
