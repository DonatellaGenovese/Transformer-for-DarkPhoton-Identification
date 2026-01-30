import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPGraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, batch):
        """
        x: node features, shape [total_nodes_in_batch, in_channels]
        batch: graph assignment, shape [total_nodes_in_batch]
               batch[i] = graph index of node i
        """
        # Node-level MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling (sum)
        num_graphs = batch.max().item() + 1
        graph_embeddings = torch.zeros(num_graphs, x.size(1), device=x.device)
        graph_embeddings = graph_embeddings.index_add_(0, batch, x)

        # Optional: could also use mean pooling:
        # counts = torch.bincount(batch)
        # graph_embeddings = graph_embeddings / counts.view(-1,1)

        # Graph-level classification
        return self.classifier(graph_embeddings)
