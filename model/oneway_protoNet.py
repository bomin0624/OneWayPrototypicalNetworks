import torch
import wandb
import utils
from data.data_loader import PrototypicalNetworkDataLoader, EvalPrototypicalNetworkDataLoader
from proto_model import PrototypicalNetwork


# pylint: disable=E1101
def main():
    utils.torch_fix_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = PrototypicalNetworkDataLoader()
    valid_loader = EvalPrototypicalNetworkDataLoader()
    # Initialize the prototypical network model
    model = PrototypicalNetwork(device)
    model.to(device)
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    max_epoch = 20
    wandb.init(
        name=f"{max_epoch}-epoch_valid",
        project="one-way prototypical network",
        config={
            "learning_rate": learning_rate,
            "architecture": "BERT",
            "epochs": max_epoch,
        }
    )

    max_valid_f1_score = 0
    for epoch in range(max_epoch):
        for iteration, batch in enumerate(data_loader):
            optimizer.zero_grad()
            loss = model(batch, device)
            loss.backward()
            optimizer.step()
            print(
                f"Epoch: {epoch + 1}, Iteration: {iteration + 1}, Loss: {loss.item()}")
            wandb.log({"loss": loss})

        valid_precision, valid_recall, valid_f1_score = model.forward_eval(
            valid_loader.get_batch(), device)
        print(
            f"Vaild Precision: {valid_precision}, Vaild Recall: {valid_recall}, Vaild F1-score: {valid_f1_score}")
        wandb.log({"vaild precision": valid_precision,
                  "vaild recall": valid_recall, "valid_f1_score": valid_f1_score})
        if valid_f1_score > max_valid_f1_score:
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, "./train_model/oneway_model.pt")
            max_valid_f1_score = valid_f1_score

    wandb.finish()


if __name__ == '__main__':
    main()
