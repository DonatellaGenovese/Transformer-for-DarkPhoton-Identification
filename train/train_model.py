import sys, os, torch, random, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torchmetrics.classification import AUROC
from torch_uncertainty.metrics.classification.calibration_error import CalibrationError, AdaptiveCalibrationError
from torch_uncertainty.metrics.classification import Entropy, BrierScore
from evaluation.metrics import uceloss
from torch.nn.functional import softmax

from train import train_evaluate, evaluate
from utils import LoadBalancingLoss
from dataset.darkphotondataset import DarkPhotonDataset
from dataset.transforms import GraphFilter
from models.model import Transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


# Settings
device = torch.device('cuda:0')
num_runs = 10
seeds = [42, 123, 456, 789, 1011, 2025, 3141, 2718, 1618, 3958239256]
#seeds = [3958239256]
bin_counts = [15, 20, 30, 50]
batchsize = 256

# Config
save_path = './results/MoE_multi_seed_datamodified'
os.makedirs(save_path, exist_ok=True)
dataset_root = '/hdd3/dongen/Desktop/darkphoton/train/data'

# Dataset
graph_filter = GraphFilter(min_num_nodes=2)
dataset = DarkPhotonDataset(
    root=dataset_root,
    subset=1.0,
    url="https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download",
    pre_filter=graph_filter,
    verbose=True
)

# Function to compute width (from dataset_statistics.ipynb)
def compute_width(x):
    """
    Compute width (max ΔR) for a graph
    x: tensor of shape [num_nodes, 4] with [layer, eta, phi, energy]
    """
    layer = x[:, 0]
    widths = []
    for l in range(4):
        mask = (layer == l)
        coords = x[mask][:, 1:3]  # [eta, phi]
        if coords.size(0) < 2:
            continue
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dR = torch.norm(diff, dim=-1)
        max_dR = dR.max().item()
        widths.append(max_dR)
    width = max(widths) if widths else 0.0
    return width


# Filter dataset to keep only graphs with width > 0
dataset_nonzero_width = [data for data in dataset if compute_width(data.x) > 0]
print(f"Original dataset size: {len(dataset)}")
print(f"Filtered dataset (width > 0): {len(dataset_nonzero_width)}")

# Use dataset_nonzero_width for training instead of dataset
dataset = dataset_nonzero_width

# Accumulators
metrics_list = []
roc_curves = []     # list of dicts: {"fpr":..., "tpr":..., "auc":...}

for run_idx, seed in enumerate(seeds):
    print(f"\n🔁 Run {run_idx + 1} with seed {seed}")

    # Reproducibility
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

    # Train/val/test split
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=seed)
    trainset, evalset = train_test_split(trainset, test_size=0.2, random_state=seed)
    
    # Create deterministic data loaders
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(seed)
    test_generator = torch.Generator()
    test_generator.manual_seed(seed)

    train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True, 
                             generator=train_generator,
                             worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    eval_loader = DataLoader(evalset, batch_size=batchsize, shuffle=False,
                            generator=eval_generator,
                            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False,
                            generator=test_generator,
                            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

    # Model and training setup
    model = Transformer(input_size=4, hidden_size=40, encoding_size=2, g_norm=False, heads=2, num_xprtz=6, xprt_size=35, k=2, dropout_encoder=0.2, layers=2, output_size=2, noise_scale=2.0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0016)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    loss_fn = LoadBalancingLoss(criterion, 3)

    # Train
    _ = train_evaluate(train_loader, eval_loader, model, criterion, loss_fn, optimizer, scheduler, 45, 80, device=device)

    # --- SAVE CHECKPOINT FOR THIS RUN ---
    checkpoint_file = os.path.join(save_path, f"checkpoint_run_{run_idx+1}.pth")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 80   # oppure sostituisci con la tua variabile epoch se lo rendi dinamico
    }, checkpoint_file)

    print(f"💾 Checkpoint salvato: {checkpoint_file}")

    # Evaluate
    result2 = evaluate(test_loader, model, loss_fn, device=device, use_noisy_routing=False)

    # Predictions
    logits = torch.cat(result2[0])
    truths = torch.cat(result2[1])
    preds = torch.argmax(logits, dim=1)
    probs_2d = softmax(logits, dim=1)
    probs_1d = probs_2d[:, 1]
    
    # --- ROC CURVE COMPUTATION ---
    y_true = truths.cpu().numpy()
    y_score = probs_1d.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    roc_curves.append({
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc
    })

    # Save each run's ROC curve
    np.savez(
        os.path.join(save_path, f"roc_run_{run_idx+1}.npz"),
        fpr=fpr, tpr=tpr, auc=roc_auc
    )

    # Metrics
    auroc_tm = AUROC(task='binary').to(device)(logits[:, 1], truths).item()
    auroc_sk = roc_auc_score(truths.cpu(), logits[:, 1].cpu())
    acc = accuracy_score(truths.cpu(), preds.cpu())
    prec = precision_score(truths.cpu(), preds.cpu())
    rec = recall_score(truths.cpu(), preds.cpu())
    f1 = f1_score(truths.cpu(), preds.cpu())
    brier = BrierScore(num_classes=2).to(device)
    brier.update(probs_2d, truths)
    brier_score = brier.compute().item()
    entropy = Entropy().to(device)
    entropy.update(probs_1d)
    entropy_val = entropy.compute().item()
    uce_val, *_ = uceloss(probs_2d, truths)

    # ECE for multiple bin settings
    ece_dict = {}
    for bins in bin_counts:
        ece = CalibrationError(task="binary", num_bins=bins).to(device)
        ace = AdaptiveCalibrationError(task="binary", num_bins=bins).to(device)
        ece.update(probs_1d, truths)
        ace.update(probs_1d, truths)
        ece_dict[f"ece_{bins}"] = ece.compute().item()
        ece_dict[f"adaptive_ece_{bins}"] = ace.compute().item()
        
    # Calcolo della purity con soglie specifiche
    thresholds = [0.8, 0.85, 0.90, 0.95, 0.99]
    purity_dict = {}

    # Verità
    is_signal = truths == 1
    is_background = truths == 0

    for thresh in thresholds:
        # Seleziono jet "taggati come segnale" con probabilità > thresh
        selected = probs_1d > thresh

        # Tra quelli selezionati, conto quanti sono segnale e quanti background
        S_sel = (is_signal & selected).sum().item()
        B_sel = (is_background & selected).sum().item()

        total_sel = S_sel + B_sel
        purity = S_sel / total_sel if total_sel > 0 else 0.0
        purity_dict[f"purity_{thresh}"] = purity


        # Collect
        # Collect and convert any tensors to floats
        metrics_list.append({
        "seed": seed,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auroc_sklearn": float(auroc_sk),
        "brier_score": float(brier_score),
        "entropy": float(entropy_val),
        "uce": float(uce_val),
        **{k: float(v) if torch.is_tensor(v) else v for k, v in ece_dict.items()},
        **{k: float(v) if torch.is_tensor(v) else v for k, v in purity_dict.items()}
    })
    
np.savez(os.path.join(save_path, "roc_all_runs.npz"), roc_curves=roc_curves)

# Convert to DataFrame
df = pd.DataFrame(metrics_list)

# Compute mean ± std
summary = df.agg(['mean', 'std']).round(6)
summary.loc['mean±std'] = summary.loc['mean'].astype(str) + " ± " + summary.loc['std'].astype(str)

# Save everything
df.to_csv(os.path.join(save_path, 'all_runs_metrics.csv'), index=False)
summary.to_csv(os.path.join(save_path, 'summary_metrics.csv'))

print(f"\n✅ Completed {num_runs} runs. Results saved to {save_path}")
print("\n📊 Summary Statistics (Mean ± Std):\n")
for column in df.columns:
    if column != "seed":
        mean_val = df[column].mean()
        std_val = df[column].std()
        print(f"{column:>25}: {mean_val:.6f} ± {std_val:.6f}")
