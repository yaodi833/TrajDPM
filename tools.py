import pickle

class Dummylogger:
    def __init__(self) -> None:
        pass
    def add_scalar(self, path, value, epoch)-> None:
        pass


def pdump(file_content, file_path):
    with open(file_path, "wb") as tar:
        pickle.dump(file_content, tar)


def pload(file_path):
    with open(file_path, "rb") as tar:
        out = pickle.load(tar)
    return out
