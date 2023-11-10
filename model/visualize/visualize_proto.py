import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data.dataset import KWDLC
from OneWayPrototypicalNetworks.model.proto_model import PrototypicalNetwork
import utils


def tsne_function(preds, labels, save_path):
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    components = tsne.fit_transform(preds)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("t-SNE analysis")

    # Add colorbar legend
    cbar = plt.colorbar()
    cbar.set_label("Label")

    # Save the plot
    plt.savefig(save_path, dpi=500)


def prepare_datasets(num_of_supports, num_of_queries, task):
    if task == "kwdlc_query":
        data = KWDLC()
    elif task == "test":
        data = KWDLC()

    # support_set = data.train_data_yes[:num_of_supports]
    # query_set = data.train_data_no[:num_of_queries]

    support_set = data.test_data_yes[:num_of_supports]
    query_set = data.test_data_no[:num_of_queries]

    return support_set, query_set


def main():
    utils.torch_fix_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_of_supports = 32
    num_of_queries = 32

    # Prepare datasets
    support_set, test_set = prepare_datasets(
        num_of_supports, num_of_queries, task="kwdlc_query"
    )

    batch = [support_set, test_set]

    # Initialize the prototypical network model
    # Use the appropriate input dimension based on your BERT model
    model = PrototypicalNetwork(device)
    model.load_state_dict(torch.load("./train_model/KWDLC_20_vaild.pt"))
    model.to(device)

    support_embeddings, query_embeddings = model.forward_eval_get_reps(batch, device)

    mean_support_embedding = support_embeddings.mean(dim=0)

    preds = torch.cat(
        (torch.stack([mean_support_embedding]), support_embeddings, query_embeddings)
    ).cpu()
    labels = [0] + [1] * num_of_queries + [2] * num_of_queries
    save_path = "./image/KWDLC_20_test.jpg"
    tsne_function(preds, labels, save_path)


if __name__ == "__main__":
    main()
