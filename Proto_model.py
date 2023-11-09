import torch
import torch.nn as nn
from torch.distributions import normal
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# Initialize the prototypical network
class PrototypicalNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model_name = 'cl-tohoku/bert-base-japanese-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        # self.linear = nn.Linear(768, 256)
        self.std = 1.0
        self.m = normal.Normal(torch.tensor([0.0]).to(
            device), torch.tensor([self.std]).to(device))

    def calculate_embeddings(self, df, device):
        sentences1 = df['sentence1'].tolist()
        sentences2 = df['sentence2'].tolist()
        sentences = [(sentence1, sentence2) for sentence1,
                     sentence2 in zip(sentences1, sentences2)]
        encoded_pair = self.tokenizer.batch_encode_plus(
            sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoded_pair['input_ids']
        input_ids = torch.tensor(input_ids).to(device)
        outputs = self.model(input_ids)
        # Get [CLS] token's embeddings
        cls_token_embeddings = outputs.pooler_output
        return cls_token_embeddings

    def forward(self, batch, device):

        support_batch, query_batch = batch

        support_embeddings = self.calculate_embeddings(support_batch, device)
        query_embeddings = self.calculate_embeddings(query_batch, device)

        # Calculate the mean embedding of the support set
        mean_support_embedding = support_embeddings.mean(dim=0)

        # Calculate the distances between mean_support_embedding and query embeddings
        distances_mean_queries = torch.norm(mean_support_embedding -
                                            query_embeddings, dim=1)

        # normal distribution
        probs = torch.exp(-(distances_mean_queries *
                          distances_mean_queries / 2.0))
        probs = torch.max(probs,
                          torch.tensor(1e-6).to(device))
        probs = torch.min(probs,
                          torch.tensor(1.0-(1e-6)).to(device))
        # print(probs)

        labels = query_batch['label']
        targets = [1.0 if label == 'yes' else 0.0 for label in labels]
        targets = torch.tensor(targets).to(device)

        # binary cross-entropy loss function
        loss = (-targets * torch.log(probs) - (1.0 - targets)
                * torch.log(1.0 - probs)).mean()

        return loss

    def forward_eval(self, batch, device):
        with torch.no_grad():
            support_batch, query_batch = batch
            support_embeddings = self.calculate_embeddings(
                support_batch, device)
            query_embeddings = self.calculate_embeddings(query_batch, device)

            mean_support_embedding = support_embeddings.mean(dim=0)

            # Calculate the distances between mean_support_embedding and query embeddings
            distances_mean_queries = torch.norm(mean_support_embedding -
                                                query_embeddings, dim=1)
            # normal distribution
            probs = torch.exp(-(distances_mean_queries *
                                distances_mean_queries / 2.0))
            probs = torch.max(probs,
                              torch.tensor(1e-6).to(device))
            probs = torch.min(probs,
                              torch.tensor(1.0-(1e-6)).to(device))

            labels = query_batch['label']
            threshold = 0.5
            tp, fp, fn = 0, 0, 0
            for prob, label in zip(probs, labels):
                predicted_label = 'yes' if prob >= threshold else 'no'
                if label == 'yes' and predicted_label == 'yes':
                    tp += 1
                elif label == 'no' and predicted_label == 'yes':
                    fp += 1
                elif label == 'yes' and predicted_label == 'no':
                    fn += 1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1_score

    def forward_eval_get_reps(self, batch, device):
        with torch.no_grad():
            support_batch, query_batch = batch
            support_embeddings = self.calculate_embeddings(
                support_batch, device)
            query_embeddings = self.calculate_embeddings(query_batch, device)
        return support_embeddings, query_embeddings
