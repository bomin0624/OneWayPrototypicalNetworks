import torch
from data.data_loader import (
    PrototypicalNetworkDataLoader,
    EvalPrototypicalNetworkDataLoader,
)
from OneWayPrototypicalNetworks.model.proto_model import PrototypicalNetwork
import utils


def main():
    utils.torch_fix_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = input("Please input the data that you want to calculate:")
    if data == "KWDLC":
        eval_loader = EvalPrototypicalNetworkDataLoader()

    # Initialize the prototypical network model
    model = PrototypicalNetwork(device)
    model.load_state_dict(torch.load("./train_model/combinepairs_0.5.pt"))
    model.to(device)

    precision, recall, f1_score = model.forward_eval(
        eval_loader.get_batch(mode="test"), device
    )

    print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")


if __name__ == "__main__":
    main()
