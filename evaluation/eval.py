# evaluate_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import random
import re
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

from dataset.darkphotondataset import DarkPhotonDataset
from dataset.transforms import GraphFilter
from models.model import Transformer
from utils import LoadBalancingLoss
from train.train import evaluate

# ---- CONFIGURATION ----
input_size = 4
hidden_size = 40
encoding_size = 2
g_norm = False
heads = 2
num_xprtz = 6
xprt_size = 35
k = 2
dropout_encoder = 0.2
layers = 2
output_size = 2
w_load = 3
batchsize = 256
seed = 3958239256
noise_scale = 2.0  
model_ckpt_path = '/home/dongen/darkphoton/results/6_35_2_0.2_3_256_0.0016_2.0_prova_seed3958239256/final_model.pt'
output_dir = './evaluation_outputs'
seeds_to_test = [seed] + [] 

os.makedirs(output_dir, exist_ok=True)

def set_seed_everything(seed):
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # For older PyTorch versions
        pass

set_seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_root = '/home/dongen/darkphoton/train/data'
os.makedirs(dataset_root, exist_ok=True)

# ---- LOAD DATA ----
graph_filter = GraphFilter(min_num_nodes=2)
dataset = DarkPhotonDataset(
    root=dataset_root,
    subset=1.0,
    url="https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download",
    pre_filter=graph_filter,
    pre_transform=None,
    post_filter=None,
    verbose=True
)

_, testset = train_test_split(dataset, test_size=0.2)
generator = torch.Generator()
generator.manual_seed(seed)
test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False, generator=generator, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

# ---- MODEL ----
model = Transformer(
    input_size, hidden_size, encoding_size, g_norm, heads,
    num_xprtz, xprt_size, k, dropout_encoder, layers, output_size
).to(device)

model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
model.eval()

criterion = torch.nn.CrossEntropyLoss()
loss = LoadBalancingLoss(criterion, 0)

for seed in seeds_to_test:
    set_seed_everything(seed)
    # ---- EVALUATION ----
    result = evaluate(test_loader, model, loss, device)
    predictions = torch.argmax(torch.cat(result[0]), dim=1)
    truths = torch.cat(result[1])
    which_node = torch.cat(result[4], dim=-1).cpu()

    # Save evaluation metrics
    acc = accuracy_score(truths.cpu(), predictions.cpu())
    precision = precision_score(truths.cpu(), predictions.cpu())
    recall = recall_score(truths.cpu(), predictions.cpu())
    test_auroc_sklearn = roc_auc_score(truths.cpu(), torch.cat(result[0])[:, 1].cpu())

    confmat = confusion_matrix(truths.cpu(), predictions.cpu())
    print(f"Accuracy: {acc:.4f}, AUROC: {test_auroc_sklearn:.4f}")

    np.savez_compressed(os.path.join(output_dir, f'eval_outputs_{seed}.npz'),
                        predictions=predictions.cpu().numpy(),
                        truths=truths.cpu().numpy(),
                        which_node=which_node.numpy(),
                        acc=acc, precision=precision, recall=recall,
                        confmat=confmat)

    # ---- METADATA FOR VISUALIZATION ----
    graph_labels = []
    graph_predictions = []  # ADD THIS LINE
    node_counts = []
    node_type_ids = []
    node_dept = []
    most_energetic_mask = []

    # Convert predictions to list for saving
    predictions_list = predictions.cpu().numpy().tolist()  # ADD THIS LINE

    for i, graph in enumerate(testset):
      label = int(graph.y.item())
      graph_labels.append(label)
      graph_predictions.append(predictions_list[i])  # ADD THIS LINE
      num_nodes = graph.x.shape[0]
      node_counts.append(num_nodes)

      node_type_ids.extend(list(range(num_nodes)))  # still dummy, per-node ids
      node_dept.extend(graph.x[:, 0].int().tolist())  # Dept is feature at index 0

      # Get the index of the most energetic node (last feature)
      energy_values = graph.x[:, -1]
      max_energy_idx = torch.argmax(energy_values).item()

      # Create binary mask: 1 for most energetic node, 0 otherwise
      mask = torch.zeros(num_nodes, dtype=torch.bool)
      mask[max_energy_idx] = 1
      most_energetic_mask.extend(mask.tolist())

    np.savez_compressed(os.path.join(output_dir, f'test_metadata_{seed}.npz'),
                    graph_labels=np.array(graph_labels, dtype=np.int32),
                    graph_predictions=np.array(graph_predictions, dtype=np.int32),  # ADD THIS LINE
                    node_counts=np.array(node_counts, dtype=np.int32),
                    node_type_ids=np.array(node_type_ids, dtype=np.int32),
                    node_dept=np.array(node_dept, dtype=np.int32),
                    most_energetic_mask=np.array(most_energetic_mask, dtype=np.bool_))
    print(f"Saved metadata for seed {seed}")

