# plot_1x3_no_pandas_logE.py
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# CSV 路径
csv_path = Path("/global/homes/x/xinyuxu/CS_230_MLIP/runs/train_log_20251110_121639.csv")

# 读取并提取需要的列（不用 pandas）
with open(csv_path, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    idx_epoch       = header.index("epoch")
    idx_train_loss  = header.index("train_loss")
    idx_train_eRMSE = header.index("train_eRMSE")
    idx_train_fRMSE = header.index("train_fRMSE")
    idx_val_loss    = header.index("val_loss")
    idx_val_eRMSE   = header.index("val_eRMSE")
    idx_val_fRMSE   = header.index("val_fRMSE")

    epoch, train_loss, val_loss = [], [], []
    train_eRMSE, val_eRMSE = [], []
    train_fRMSE, val_fRMSE = [], []

    for row in reader:
        if not row:
            continue
        epoch.append(int(row[idx_epoch]))
        train_loss.append(float(row[idx_train_loss]))
        val_loss.append(float(row[idx_val_loss]))
        train_eRMSE.append(float(row[idx_train_eRMSE]))
        val_eRMSE.append(float(row[idx_val_eRMSE]))
        train_fRMSE.append(float(row[idx_train_fRMSE]))
        val_fRMSE.append(float(row[idx_val_fRMSE]))

# 一行三列：(1) loss(train/val), (2) eRMSE(train/val, log y), (3) fRMSE(train/val)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=140, sharex=True)

# 图1：loss
ax = axes[0]
ax.plot(epoch, train_loss, label="train_loss", linewidth=2)
ax.plot(epoch, val_loss,   label="val_loss",   linewidth=2)
ax.set_title("Loss (train vs val)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.grid(True, linestyle="--", alpha=0.4); ax.legend()

# 图2：eRMSE（log y）
ax = axes[1]
ax.plot(epoch, train_eRMSE, label="train_eRMSE", linewidth=2)
ax.plot(epoch, val_eRMSE,   label="val_eRMSE",   linewidth=2)
ax.set_yscale("log")  # E 的曲线用对数坐标
ax.set_title("Energy RMSE (train vs val, log y)")
ax.set_xlabel("Epoch"); ax.set_ylabel("eRMSE")
ax.grid(True, which="both", linestyle="--", alpha=0.4); ax.legend()

# 图3：fRMSE
ax = axes[2]
ax.plot(epoch, train_fRMSE, label="train_fRMSE", linewidth=2)
ax.plot(epoch, val_fRMSE,   label="val_fRMSE",   linewidth=2)
ax.set_title("Force RMSE (train vs val)")
ax.set_xlabel("Epoch"); ax.set_ylabel("fRMSE")
ax.grid(True, linestyle="--", alpha=0.4); ax.legend()

fig.tight_layout()
out_path = csv_path.parent / "fig_1x3_loss_eRMSE_fRMSE.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"[Saved] {out_path}")
