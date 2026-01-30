import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from torch_uncertainty.metrics.classification.brier_score import BrierScore
from torch_uncertainty.metrics.classification.calibration_error import CalibrationError, AdaptiveCalibrationError
from torch_uncertainty.metrics.classification import Entropy
from evaluation.metrics import uceloss
from utils import enable_dropout, check_dropout_active



# Early stopping class to prevent overfitting and stop training when validation loss stops improving
class EarlyStopping():
    def __init__(self, patience, delta=0, path='checkpoint.pt'):
        self.patience = patience  # Number of epochs to wait before stopping after no improvement
        self.delta = delta  # Minimum change in loss to qualify as an improvement
        self.path = path  # Path to save the best model
        self.counter = 0  # Counts the number of epochs with no improvement
        self.best_score = None  # Tracks the best validation score
        self.early_stop = False  # Flag to indicate if training should stop

    def __call__(self, val_loss, model):
        score = -val_loss  # Convert loss to a score (minimization problem)
        
        # Initialize best score in the first call
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            # No improvement, increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training
        else:
            # Improvement found, save model and reset counter
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        print(f'Model saved with validation loss: {val_loss}')


# Training function
def train(dataloader, model, loss_fn, optimizer, device, profiler=None):
    model.train()
    predictions = []
    ground_truths = []
    losses = []
    expert_sample_counts = []
    expert_loads = []

    for batch in dataloader:
        optimizer.zero_grad()
        prediction = model(batch.to(device))
        batch_loss = loss_fn(prediction, batch.y)

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        predictions.append(prediction[0])
        ground_truths.append(batch.y)
        losses.append(batch_loss)
        expert_sample_counts.append(prediction[2])
        expert_loads.append(prediction[1])

        if profiler is not None:
            profiler.step()

    return predictions, ground_truths, expert_loads, expert_sample_counts



# Evaluation function (similar to training but without gradient updates)
# Modified evaluate with profiler support
def evaluate(dataloader, model, loss_fn, device, use_noisy_routing=False, en_dropout=False, profiler=None):
    model.eval()
    batch_count = 0
    predictions = []
    ground_truths = []
    losses = []
    expert_loads = []
    expert_sample_counts = []
    node_expert_assignments_list = []
    routing_logits = []
    all_node_type_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if en_dropout:
                enable_dropout(model)
                check_dropout_active(model)

            prediction = model(batch.to(device), use_noisy_routing=use_noisy_routing)
            batch_loss = loss_fn(prediction, batch.y)

            predictions.append(prediction[0])
            ground_truths.append(batch.y)
            losses.append(batch_loss)
            expert_loads.append(prediction[1])
            expert_sample_counts.append(prediction[2])
            node_expert_assignments_list.append(prediction[3])
            routing_logits.append(prediction[4])

            all_node_type_ids.extend(range(batch.x.size(0)))

            batch_count += 1

            if profiler is not None:
                profiler.step()

    return predictions, ground_truths, expert_loads, expert_sample_counts, node_expert_assignments_list, routing_logits, all_node_type_ids

# Training and evaluation loop
def train_evaluate(trainloader, evalloader, model, criterion, loss_fn, optimizer, scheduler, patience, epochs, device, save_dir="training_outputs"):
    os.makedirs(save_dir, exist_ok=True)  # Create a folder to save outputs

    train_history, eval_history = [], []
    early_stopping = EarlyStopping(patience=epochs)  # Initialize early stopping
    scheduler_wait_counter = 0

    # Metrics tracking
    train_accuracy_list, train_precision_list, train_recall_list, train_load_balance_list = [], [], [], []
    eval_accuracy_list, eval_precision_list, eval_recall_list, eval_load_balance_list = [], [], [], []

    # Uncertainty & calibration metrics (for eval only)
    entropy_metric = Entropy().to(device)
    ece_metric = CalibrationError(task="binary",num_bins=15).to(device)
    adaptive_ece_metric = AdaptiveCalibrationError(task="binary",num_bins=15).to(device)
    brier_metric = BrierScore(num_classes=2, reduction="mean").to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        # Training
        train_results = train(dataloader=trainloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        # Evaluation
        eval_results = evaluate(dataloader=evalloader, model=model, loss_fn=loss_fn, device=device)
        # Store history
        train_history.append(train_results)
        eval_history.append(eval_results)

        # Learning rate scheduling (if applicable)
        if scheduler is not None:
            if scheduler_wait_counter == patience:
                model.load_state_dict(torch.load('checkpoint.pt'))  # Load best model
                print("Best model loaded before stepping")
            scheduler.step()
        scheduler_wait_counter += 1

        # Compute losses for training and evaluation
        train_loss = criterion(torch.cat(train_results[0]), torch.cat(train_results[1]))
        eval_loss = criterion(torch.cat(eval_results[0]), torch.cat(eval_results[1]))

        # Compute accuracy/precision/recall for train and eval
        train_preds_epoch = torch.argmax(torch.cat(train_results[0]), dim=1)
        eval_preds_epoch = torch.argmax(torch.cat(eval_results[0]), dim=1)
        
        # Compute classification metrics
        train_accuracy_list.append(accuracy_score(torch.cat(train_results[1]).cpu(), train_preds_epoch.cpu()))
        eval_accuracy_list.append(accuracy_score(torch.cat(eval_results[1]).cpu(), eval_preds_epoch.cpu()))

        train_precision_list.append(precision_score(torch.cat(train_results[1]).cpu(), train_preds_epoch.cpu()))
        eval_precision_list.append(precision_score(torch.cat(eval_results[1]).cpu(), eval_preds_epoch.cpu()))

        train_recall_list.append(recall_score(torch.cat(train_results[1]).cpu(), train_preds_epoch.cpu()))
        eval_recall_list.append(recall_score(torch.cat(eval_results[1]).cpu(), eval_preds_epoch.cpu()))

        # Load balancing stats
        train_load_balance_list.append(torch.mean(torch.tensor(train_results[3])))
        eval_load_balance_list.append(torch.mean(torch.tensor(eval_results[3])))

        # --- New: compute uncertainty & calibration metrics on eval set ---
        entropy_metric.reset()
        ece_metric.reset()
        adaptive_ece_metric.reset()
        brier_metric.reset()

        logits = torch.cat(eval_results[0])
        labels = torch.cat(eval_results[1])
        probs_2d = torch.nn.functional.softmax(logits, dim=1)  # shape [N, 2]
        probs_1d = probs_2d[:, 1]  # shape [N]



        entropy_metric.update(probs_1d)
        ece_metric.update(probs_1d, labels)
        adaptive_ece_metric.update(probs_1d, labels)
        brier_metric.update(probs_2d, labels)

        entropy_val = entropy_metric.compute()
        ece_val = ece_metric.compute()
        adaptive_ece_val = adaptive_ece_metric.compute()
        brier_val = brier_metric.compute()

        uce_val, err_in_bin, avg_entropy_in_bin = uceloss(probs_2d, labels)  # your custom UCE function

        # Print epoch summary
        print(f"Epoch {epoch + 1}")
        print(f"Training loss: {train_loss:.4f} | Eval loss: {eval_loss:.4f}")
        print(f"Train Acc: {train_accuracy_list[-1]:.4f} | Eval Acc: {eval_accuracy_list[-1]:.4f}")
        print(f"Train Prec: {train_precision_list[-1]:.4f} | Eval Prec: {eval_precision_list[-1]:.4f}")
        print(f"Train Recall: {train_recall_list[-1]:.4f} | Eval Recall: {eval_recall_list[-1]:.4f}")
        print(f"Train Load Bal: {train_load_balance_list[-1]:.4f} | Eval Load Bal: {eval_load_balance_list[-1]:.4f}")
        print(f"Entropy: {entropy_val.item():.4f} | ECE: {ece_val.item():.4f} | Adaptive ECE: {adaptive_ece_val.item():.4f}")
        print(f"UCE: {uce_val.item():.4f} | Brier Score: {brier_val.item():.4f}")
        print("-" * 60)

        # Early stopping check
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    # Save outputs for plotting/analysis
    torch.save({
        "train_expert_sample_counts": train_results[3],
        "eval_expert_sample_counts": eval_results[3],
        "eval_node_expert_assignments": eval_results[4],
    }, os.path.join(save_dir, "experts_info.pt"))

    torch.save({
        "train_accuracy_list": train_accuracy_list,
        "train_precision_list": train_precision_list,
        "train_recall_list": train_recall_list,
        "train_load_balance_list": train_load_balance_list,
        "eval_accuracy_list": eval_accuracy_list,
        "eval_precision_list": eval_precision_list,
        "eval_recall_list": eval_recall_list,
        "eval_load_balance_list": eval_load_balance_list,
    }, os.path.join(save_dir, "metrics.pt"))

    return train_history, eval_history, (train_accuracy_list, train_precision_list, train_recall_list, train_load_balance_list), (eval_accuracy_list, eval_precision_list, eval_recall_list, eval_load_balance_list)
