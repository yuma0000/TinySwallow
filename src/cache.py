class cache:
    def __init__(path: str):
        self.k = []
        self.v = []

    def search():

    def get():

    def save():

    def add_kv(k, v):
        self.k = F.concatenate(self.k, k, axis=2)
        self.v = F.concatenate(self.v, v, axis=2)
        return self.k, self.v

    def delete_kv(c):
        self.k = F.slice(self.k, start=(0,0,0,0), stop=(self.k.shape[0], self.k.shape[1], self.k.shape[2] - c, self.k.shape[3]))
        self.v = F.slice(self.v, start=(0,0,0,0), stop=(self.v.shape[0], self.v.shape[1], self.v.shape[2] - c, self.v.shape[3]))
