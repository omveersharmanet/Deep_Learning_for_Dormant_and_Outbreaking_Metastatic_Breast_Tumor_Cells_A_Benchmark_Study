import os
import torch
import torch.optim as optim
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for saving plots
import matplotlib.pyplot as plt

import logging
from utils.logger import setup_logger, get_eval_logger
import pickle
import scipy.io
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import math




def train_model(
    model, 
    train_loader,   
    val_loader,   
    test_loader,          
    epochs, 
    lr, 
    device, 
    logger, 
    exp_dir, 
    args=None,
    patience_iter=5,    # patience before reducing LR
    early_stop_iter=20  # early stop if no improvement
):
    

    if args is not None:
        logger.info(f"Using batch_size={args.batch_size}")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float("inf")
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    val_no_improve = 0
    lr_patience = 0


    # =======================================================
    # Start training
    # =======================================================
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0
        for xb, yb in train_loader:
            xb, yb = xb.squeeze(1).to(device), yb.to(device)
            #print('train',xb.shape)
            #print(xb.shape,yb.shape)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
        
        # ---------------------------------------------------
        # Validation
        # ---------------------------------------------------
        model.eval()
        val_loss_sum = 0
        val_correct = defaultdict(int)
        val_total = defaultdict(int)
        overall_correct = 0
        overall_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.squeeze(1).to(device), yb.to(device)
                #print('val',xb.shape)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss_sum += loss.item() * xb.size(0)

                preds = outputs.argmax(1)
                overall_correct += (preds == yb).sum().item()
                overall_total += yb.numel()

                for cls in yb.unique():
                    mask = yb == cls
                    val_correct[int(cls)] += (preds[mask] == yb[mask]).sum().item()
                    val_total[int(cls)] += mask.sum().item()
        
        val_loss = val_loss_sum / len(val_loader.dataset)
        overall_acc = 100 * overall_correct / overall_total
        class_acc = {cls: 100 * val_correct[cls] / val_total[cls] for cls in val_correct}

        logger.info(f"Epoch {epoch} | Train Loss: {train_loss_sum/len(train_loader.dataset):.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {overall_acc:.3f}% | Class Acc: {class_acc}")

        # ---------------------------------------------------
        # Scheduler & early stopping
        # ---------------------------------------------------
        #if val_loss < best_val_loss:
        if ((1 - overall_acc/100) + sum(1 - acc/100 for acc in class_acc.values()) / len(class_acc)) / 3 < best_val_loss:
        #if ((1 - overall_acc/100) + sum(1 - acc/100 for acc in class_acc.values())) / 3 < best_val_loss:        
            #best_val_loss = val_loss
            best_val_loss = ((1 - overall_acc/100) + sum(1 - acc/100 for acc in class_acc.values()) / len(class_acc)) / 3 
            #best_val_loss = ((1 - overall_acc/100) + sum(1 - acc/100 for acc in class_acc.values()) ) / 3 
            torch.save(model.state_dict(), best_model_path)
            val_no_improve = 0
            lr_patience = 0
            logger.info(f"Saved new best model at epoch {epoch}")
        else:
            val_no_improve += 1
            lr_patience += 1

        if lr_patience >= patience_iter:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.5
                logger.info(f"Reducing LR from {old_lr:.6f} to {param_group['lr']:.6f}")
            lr_patience = 0

        if val_no_improve >= early_stop_iter:
            logger.info(f"No improvement for {early_stop_iter} epochs. Stopping training early.")
            break

        scheduler.step(val_loss)#scheduler.step()
    
    logger.info(f"Training finished. Best model saved at: {best_model_path}")


    # ==========================================================
    # Load BEST model 
    # ==========================================================
    logger.info("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ----------------------------------------------------------
    # Plot: Overall accuracy
    # ----------------------------------------------------------


    test_correct = defaultdict(int)
    test_total = defaultdict(int)
    test_overall_correct = 0
    test_overall_total = 0

    all_outputs = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.squeeze(1).to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(1)
            test_overall_correct += (preds == yb).sum().item()
            test_overall_total += yb.numel()
            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())
            for cls in yb.unique():
                mask = yb == cls
                test_correct[int(cls)] += (preds[mask] == yb[mask]).sum().item()
                test_total[int(cls)] += mask.sum().item()

    outputs = torch.cat(all_outputs)     # [N, C]
    preds   = torch.cat(all_preds)       # [N]
    targets = torch.cat(all_targets)     # [N]
        
    test_overall_acc = 100 * test_overall_correct / test_overall_total
    test_class_acc = {cls: 100 * test_correct[cls] / test_total[cls] for cls in test_correct}
    logger.info(f"Test Acc: {test_overall_acc:.3f}% | Test Class Acc: {test_class_acc}")
    
    cm = confusion_matrix(targets.numpy(), preds.numpy())

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Epoch {epoch})")

    cm_path = os.path.join(exp_dir, f"best_model_confusion_matrix_.png")
    plt.savefig(cm_path)
    plt.close()
    save_path = os.path.join(exp_dir, f"best_model_test_results_.pt")
    torch.save({
        "outputs": outputs,     # logits
        "preds": preds,         # predicted labels
        "targets": targets,     # ground truth
        "confusion_matrix": cm  # saved CM
    }, save_path)

    if outputs.shape[1] == 2:   # binary only
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()

        fpr, tpr, _ = roc_curve(targets.numpy(), probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Test)")
        plt.legend()
        plt.grid(True)

        roc_path = os.path.join(exp_dir, f"best_model_roc_.png")
        plt.savefig(roc_path)
        plt.close()

        logger.info(f"ROC AUC: {roc_auc:.4f}")


def evaluate_best_model(
    model,
    val_loader,
    test_loader,
    device,
    exp_dir,
    args=None,
):
    eval_logger = get_eval_logger(exp_dir)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    eval_logger.info("===== EVALUATION STARTED =====")



    best_model_path = os.path.join(exp_dir, "best_model.pth")

    assert os.path.exists(best_model_path), \
        f"Best model not found: {best_model_path}"

    eval_logger.info(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()




    criterion = torch.nn.CrossEntropyLoss()

    # =====================================================
    # Validation evaluation
    # =====================================================
    val_loss_sum = 0
    val_correct = defaultdict(int)
    val_total = defaultdict(int)
    val_overall_correct = 0
    val_overall_total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.squeeze(1).to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)

            val_loss_sum += loss.item() * xb.size(0)
            preds = outputs.argmax(1)

            val_overall_correct += (preds == yb).sum().item()
            val_overall_total += yb.numel()

            for cls in yb.unique():
                mask = yb == cls
                val_correct[int(cls)] += (preds[mask] == yb[mask]).sum().item()
                val_total[int(cls)] += mask.sum().item()

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_overall_acc = 100 * val_overall_correct / val_overall_total
    val_class_acc = {
        cls: 100 * val_correct[cls] / val_total[cls]
        for cls in val_correct
    }




    eval_logger.info("[VALIDATION RESULTS]")
    eval_logger.info(f"Loss: {val_loss:.4f}")
    eval_logger.info(f"Overall Val Acc: {val_overall_acc:.3f}%")
    eval_logger.info(f"Class-wise Acc: {val_class_acc}")

    # =====================================================
    # Test evaluation (overall + class)
    # =====================================================
    test_correct = defaultdict(int)
    test_total = defaultdict(int)
    test_overall_correct = 0
    test_overall_total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.squeeze(1).to(device), yb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(1)

            test_overall_correct += (preds == yb).sum().item()
            test_overall_total += yb.numel()

            for cls in yb.unique():
                mask = yb == cls
                test_correct[int(cls)] += (preds[mask] == yb[mask]).sum().item()
                test_total[int(cls)] += mask.sum().item()

    test_overall_acc = 100 * test_overall_correct / test_overall_total
    test_class_acc = {
        cls: 100 * test_correct[cls] / test_total[cls]
        for cls in test_correct
    }

    eval_logger.info("[TEST RESULTS]")
    eval_logger.info(f"Overall test Acc: {test_overall_acc:.3f}%")
    eval_logger.info(f"Class-wise Acc: {test_class_acc}")

    eval_logger.info("Evaluation completed successfully.")
    