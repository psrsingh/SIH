"""Fine-tune Depth Anything Small on GAMUS with held-out validation."""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GamusDepthDataset
from metrics import regression_metrics


def ssi_loss(prediction, target, valid):
    losses = []
    for pred, truth, mask in zip(prediction, target, valid):
        mask = mask.squeeze(0)
        x, y = pred.squeeze(0)[mask], truth.squeeze(0)[mask]
        if x.numel() < 8:
            continue
        A = torch.stack([x, torch.ones_like(x)], dim=1)
        scale, shift = torch.linalg.lstsq(A, y.unsqueeze(1)).solution[:2, 0]
        residual = scale * x + shift - y
        losses.append(torch.mean(residual ** 2) / (torch.mean(y ** 2) + 1e-6))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def gradient_loss(prediction, target, valid):
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_x = target[..., :, 1:] - target[..., :, :-1]
    true_y = target[..., 1:, :] - target[..., :-1, :]
    mask_x = valid[..., :, 1:] & valid[..., :, :-1]
    mask_y = valid[..., 1:, :] & valid[..., :-1, :]
    values = []
    if mask_x.any(): values.append(torch.abs(pred_x[mask_x] - true_x[mask_x]).mean())
    if mask_y.any(): values.append(torch.abs(pred_y[mask_y] - true_y[mask_y]).mean())
    return torch.stack(values).mean() if values else prediction.sum() * 0.0


def _seed_everything(seed):
    random.seed(seed)
    np = __import__("numpy")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _normalise(images, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    return (images - mean) / std


def _run_validation(model, loader, device):
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for batch in loader:
            images = _normalise(batch["pixel_values"].to(device), device)
            target = batch["depth"].to(device)
            valid = batch["valid"].to(device)
            prediction = model(pixel_values=images).predicted_depth.unsqueeze(1)
            prediction = F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss = ssi_loss(prediction, target, valid) + 0.5 * gradient_loss(prediction, target, valid)
            losses.append(float(loss))
            metrics.append(regression_metrics(prediction[0, 0].cpu().numpy(), target[0, 0].cpu().numpy(), valid[0, 0].cpu().numpy()))
    model.train()
    if not metrics:
        raise ValueError("Validation split produced no valid samples")
    return {
        "loss": sum(losses) / len(losses),
        **{key: sum(item[key] for item in metrics) / len(metrics) for key in ("rmse", "mae", "correlation")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("backend/models/depth-anything-gamus"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    _seed_everything(args.seed)
    from transformers import AutoModelForDepthEstimation
    model_name = str(args.resume) if args.resume else "LiheYoung/depth-anything-small-hf"
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).train()
    train_data = GamusDepthDataset("train", root=args.data_root, crop_size=args.crop_size, max_records=args.max_records, cache_dir=args.cache_dir)
    val_data = GamusDepthDataset("validation", root=args.data_root, crop_size=args.crop_size, augment=False, max_records=args.max_records, cache_dir=args.cache_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train.csv"
    best_path = args.output / "best"
    latest_path = args.output / "latest.pt"
    best_rmse = float("inf")
    start_epoch = 0
    if args.resume and Path(args.resume).is_file():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "rmse", "mae", "correlation"])
        if handle.tell() == 0: writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            train_losses = []
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader):
                images = batch["pixel_values"].to(device)
                images = _normalise(images, device)
                target = batch["depth"].to(device)
                valid = batch["valid"].to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    output = model(pixel_values=images).predicted_depth.unsqueeze(1)
                    output = F.interpolate(output, size=target.shape[-2:], mode="bilinear", align_corners=False)
                    loss_ssi = ssi_loss(output, target, valid)
                    loss_grad = gradient_loss(output, target, valid)
                    loss = (loss_ssi + 0.5 * loss_grad) / max(args.grad_accumulation, 1)
                scaler.scale(loss).backward()
                if (step + 1) % max(args.grad_accumulation, 1) == 0 or step + 1 == len(train_loader):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                train_losses.append(float(loss.detach()) * max(args.grad_accumulation, 1))
            validation = _run_validation(model, val_loader, device)
            train_loss = sum(train_losses) / max(len(train_losses), 1)
            writer.writerow({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation["loss"],
                             "rmse": validation["rmse"], "mae": validation["mae"], "correlation": validation["correlation"]})
            handle.flush()
            state = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": args.seed, "validation": validation}
            torch.save(state, latest_path)
            if validation["rmse"] < best_rmse:
                best_rmse = validation["rmse"]
                model.save_pretrained(best_path)
                (args.output / "best_metrics.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
            print(f"epoch={epoch + 1} train={train_loss:.5f} val_rmse={validation['rmse']:.5f}")
    model.save_pretrained(args.output / "final")
    print(f"saved latest checkpoint to {latest_path}")
    print(f"saved best checkpoint to {best_path}")


if __name__ == "__main__": main()
