import argparse
import torch
import os
from datetime import datetime

from data.dataloader_2dn import load_2d_data_seq


from train.trainer import train_model, evaluate_best_model
from utils.logger import setup_logger
from models import get_model_registry






def main(args):
    # -------------------------
    MODEL_REGISTRY = get_model_registry(args.framesize)
    #model = MODEL_REGISTRY[args.model]()
    class_dirs = {
        "class1_1st_data": 0,
        "class_2_3rddata": 1
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Channels string (for experiment naming)
    # -------------------------
    channels_str = "-".join(str(c) for c in args.channels)

    # -------------------------
    # Data
    # -------------------------

    train_loader, val_loader, test_loader = load_2d_data_seq(
        root=args.root,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        framesize=args.framesize
    ) 
    
    model_class = MODEL_REGISTRY[args.modelname]

    # ✅ SINGLE, CONSISTENT CONTRACT
    model = model_class(
        args=args,
        num_classes=len(class_dirs)
    ).to(device)
    args.results_dir = (f"{args.results_dir}/sequence_length{args.sequence_length}/{args.framesize}x{args.framesize}")
    print(model)

    # -------------------------
    # Train / Evaluate
    # -------------------------
    if args.is_traning == 1:
        for ii in range(args.itr):
            exp_name = (
                f"Size{args.framesize}x" 
                f"{args.framesize}_"  
                f"sequence_length{args.sequence_length}_"                             
                f"layers{args.num_layers}_"
                f"c{channels_str}_"
                f"dropout{args.dropout}_"
                f"itr{ii}"
            )

            exp_dir = os.path.join(args.results_dir, args.modelname, exp_name)
            best_model_path = os.path.join(exp_dir, "best_model.pth")
            final_roc_plot = os.path.join(exp_dir, "best_model_roc_.png")
            # 1. Check if the entire experiment is already finished (Plots exist)
            if os.path.exists(final_roc_plot):
                print(f"SKIPPING: {exp_name} is already complete.")
                continue
            # 2. Setup directory and logger
            os.makedirs(exp_dir, exist_ok=True)
            logger = setup_logger(exp_dir)
            logger.info(vars(args))
            # 3. Check for existing best_model.pth to load weights
            if os.path.exists(best_model_path):
                logger.info(f"Existing model found at {best_model_path}. Loading weights to resume training.")
                model.load_state_dict(torch.load(best_model_path, map_location=device))
            else:
                logger.info("No existing model found. Starting training from scratch.")
            # 4. Run the training function (this will now start from loaded weights if they existed)
            logger.info("Training started")


            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                logger=logger,
                exp_dir=exp_dir,
                args=args,
                patience_iter=args.patience_iter,
                early_stop_iter=args.early_stop_iter
            )

    elif args.is_traning == 2:
        for ii in range(args.itr):
            exp_name = (
                f"Size{args.framesize}x" 
                f"{args.framesize}_"  
                f"sequence_length{args.sequence_length}_"                             
                f"layers{args.num_layers}_"
                f"c{channels_str}_"
                f"dropout{args.dropout}_"
                f"itr{ii}"
            )

            exp_dir = os.path.join(args.results_dir, args.modelname, exp_name)    
            os.makedirs(exp_dir, exist_ok=True)

            logger = setup_logger(exp_dir)
            logger.info("Evaluation started")
            logger.info(vars(args))

            evaluate_best_model(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                exp_dir=exp_dir,
                args=args
            )


    elif args.is_traning == 0:
        ii = args.itr
        exp_name = (
            f"Size{args.framesize}x" 
            f"{args.framesize}_"  
            f"sequence_length{args.sequence_length}_"                             
            f"layers{args.num_layers}_"
            f"c{channels_str}_"
            f"dropout{args.dropout}_"
            f"itr{ii}"
        )

        exp_dir = os.path.join(args.results_dir, args.modelname, exp_name)

        evaluate_best_model(
            model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            exp_dir=exp_dir,
            args=args
        )

    else:
        raise ValueError("Invalid --is_traning value. Use 0, 1, or 2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--results_dir", required=True)

    parser.add_argument("--modelname", required=True)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_lastlayerstrain", type=int, default=1)
    parser.add_argument("--channels", nargs="+", type=int,default=1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--normalize", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience_iter", type=int, default=5)
    parser.add_argument("--early_stop_iter", type=int, default=20)
    parser.add_argument("--is_traning", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--framesize", type=int, default=96)
    # Generic extra args (safe for all models)
    parser.add_argument("--hidden_dim", type=int, default=128)




    args = parser.parse_args()
    main(args)
