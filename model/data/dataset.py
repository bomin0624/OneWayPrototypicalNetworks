import pandas as pd
from sklearn.model_selection import train_test_split


class KWDLC:
    def __init__(self):
        self.data = pd.read_csv(
            "/home_lab_local/s2110410/research/dataset/trainingdata/KWDLC-R.csv"
        )
        self.data_yes = self.data[self.data["label"] == "yes"]
        self.data_no = self.data[self.data["label"] == "no"]

        self.train_data_yes, self.val_data_yes = train_test_split(
            self.data_yes, train_size=0.8, test_size=0.2, random_state=42
        )
        self.train_data_no, self.val_data_no = train_test_split(
            self.data_no, train_size=0.8, test_size=0.2, random_state=42
        )

        self.val_data_yes, self.test_data_yes = train_test_split(
            self.val_data_yes, test_size=0.5, random_state=42
        )
        self.val_data_no, self.test_data_no = train_test_split(
            self.val_data_no, test_size=0.5, random_state=42
        )
