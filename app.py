import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # prevents display errors on some systems
import seaborn as sns
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────────────
# 1. SETUP
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "dataset"   # change this if your folder is named differently
EPOCHS    = 10
BATCH     = 32
LR        = 0.001
SEED      = 42

# ──────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMS
#    - ImageNet normalisation (mean/std) is REQUIRED for pre-trained ResNet
#    - Training set gets augmentation to improve robustness
#    - Val/Test sets get NO augmentation (unbiased evaluation)
# ──────────────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),          # augmentation
    transforms.RandomRotation(15),              # augmentation
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2),       # augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std =[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────────────────────────────────────────
# 3. LOAD DATASET AND SPLIT 70 / 15 / 15
# ──────────────────────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
classes      = full_dataset.classes
num_classes  = len(classes)
total        = len(full_dataset)

train_size = int(0.70 * total)
val_size   = int(0.15 * total)
test_size  = total - train_size - val_size

torch.manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full_dataset,
                                         [train_size, val_size, test_size])

# Apply correct transforms to val and test
# (random_split shares the parent dataset transform, so we wrap them)
val_ds.dataset  = datasets.ImageFolder(root=data_dir, transform=val_test_transform)
test_ds.dataset = datasets.ImageFolder(root=data_dir, transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

print(f"\nClasses : {classes}")
print(f"Total   : {total} images")
print(f"Train   : {len(train_ds)}")
print(f"Val     : {len(val_ds)}")
print(f"Test    : {len(test_ds)}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL  (Transfer Learning with ResNet18)
#    - Load pre-trained weights from ImageNet  →  Pre-Training (PT)
#    - Freeze all conv layers
#    - Replace final FC layer with one for our 6 classes  →  Fine-Tuning (FT)
# ──────────────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze ALL pre-trained layers (feature extraction mode)
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier  (this is the only layer that will train)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

print(f"Model   : ResNet18 (transfer learning)")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. LOSS & OPTIMISER
# ──────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP  (with validation each epoch)
# ──────────────────────────────────────────────────────────────────────────────
train_losses = []
val_losses   = []
val_accs     = []
best_val_loss = float('inf')

print("=" * 55)
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>10}")
print("=" * 55)

for epoch in range(EPOCHS):

    # ── Training ──────────────────────────────────────────
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ── Validation ────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    correct  = 0
    total_v  = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct  += (predicted == labels).sum().item()
            total_v  += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total_v

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accs.append(val_accuracy)

    # Save best model
    marker = ""
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_waste_model.pth")
        marker = "  ← best saved"

    print(f"{epoch+1:>6} {avg_train_loss:>12.4f} {avg_val_loss:>10.4f} "
          f"{val_accuracy:>9.2f}%{marker}")

print("=" * 55)
print("\nTraining complete!\n")

# ──────────────────────────────────────────────────────────────────────────────
# 7. GRAPH 1 — Training vs Validation Loss Curve
# 
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(range(1, EPOCHS+1), train_losses, 'b-o', label='Training Loss',   linewidth=2)
plt.plot(range(1, EPOCHS+1), val_losses,   'r-o', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Loss',  fontsize=13)
plt.title('Training vs Validation Loss (ResNet18 Transfer Learning)', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(range(1, EPOCHS+1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure6_loss_curve.png', dpi=150)
plt.close()
print("Saved: figure6_loss_curve.png")

# ──────────────────────────────────────────────────────────────────────────────
# 8. GRAPH 2 — Validation Accuracy over Epochs
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(range(1, EPOCHS+1), val_accs, 'g-o', linewidth=2)
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Validation Accuracy (%)', fontsize=13)
plt.title('Validation Accuracy over Training Epochs', fontsize=14)
plt.xticks(range(1, EPOCHS+1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure_val_accuracy.png', dpi=150)
plt.close()
print("Saved: figure_val_accuracy.png")

# ──────────────────────────────────────────────────────────────────────────────
# 9. TEST SET EVALUATION  (load the BEST saved model)
# ──────────────────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load("best_waste_model.pth", map_location=device))
model.eval()

all_preds  = []
all_labels = []
all_confs  = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs   = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(conf.cpu().numpy())

# ── Overall accuracy ──────────────────────────────────────────────────────────
accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

print("\n" + "=" * 55)
print("  TEST SET RESULTS")
print("=" * 55)
print(f"  Overall Accuracy: {accuracy:.2f}%")
print("=" * 55)

# ── Classification report (precision / recall / F1) ──────────────────────────
report = classification_report(all_labels, all_preds, target_names=classes)
print("\nClassification Report:")
print(report)

# Save the report to a text file so you can copy numbers into your report
with open("classification_report.txt", "w") as f:
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
    f.write(report)
print("Saved: classification_report.txt")

# ── Raw confusion matrix (numbers for your Table 5) ──────────────────────────
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix (raw numbers for Table 5):")
print("Rows = Actual,  Columns = Predicted")
print(f"{'':12}", end="")
for c in classes:
    print(f"{c:12}", end="")
print()
for i, row in enumerate(cm):
    print(f"{classes[i]:12}", end="")
    for val in row:
        print(f"{val:12}", end="")
    print()

# ──────────────────────────────────────────────────────────────────────────────
# 10. GRAPH 3 — Confusion Matrix Heatmap  (Figure 7 in your report)
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, linecolor='grey')
plt.xlabel('Predicted Label', fontsize=13)
plt.ylabel('Actual Label',    fontsize=13)
plt.title('Confusion Matrix — Test Set', fontsize=14)
plt.tight_layout()
plt.savefig('figure7_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: figure7_confusion_matrix.png")

# ──────────────────────────────────────────────────────────────────────────────
# 11. GRAPH 4 — Confidence Score Distribution  (Figure 11 in your report)
# ──────────────────────────────────────────────────────────────────────────────
all_preds_arr  = np.array(all_preds)
all_labels_arr = np.array(all_labels)
all_confs_arr  = np.array(all_confs)

correct_confs   = all_confs_arr[all_preds_arr == all_labels_arr]
incorrect_confs = all_confs_arr[all_preds_arr != all_labels_arr]

plt.figure(figsize=(9, 5))
plt.hist(correct_confs,   bins=20, alpha=0.6, color='green', label=f'Correct   (n={len(correct_confs)})')
plt.hist(incorrect_confs, bins=20, alpha=0.6, color='red',   label=f'Incorrect (n={len(incorrect_confs)})')
plt.xlabel('Confidence Score', fontsize=13)
plt.ylabel('Number of Predictions', fontsize=13)
plt.title('Confidence Score Distribution — Correct vs Incorrect Predictions', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure11_confidence_distribution.png', dpi=150)
plt.close()
print("Saved: figure11_confidence_distribution.png")

# ──────────────────────────────────────────────────────────────────────────────
# 12. GRAPH 5 — Per-Class F1 Score Bar Chart  
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import f1_score, precision_score, recall_score

f1s        = f1_score(all_labels, all_preds, average=None)
precisions = precision_score(all_labels, all_preds, average=None)
recalls    = recall_score(all_labels, all_preds, average=None)

x = np.arange(num_classes)
w = 0.25

plt.figure(figsize=(11, 6))
plt.bar(x - w, precisions, w, label='Precision', color='steelblue')
plt.bar(x,     recalls,    w, label='Recall',    color='darkorange')
plt.bar(x + w, f1s,        w, label='F1-Score',  color='seagreen')
plt.xticks(x, classes, fontsize=12)
plt.ylabel('Score', fontsize=13)
plt.title('Per-Class Performance Metrics (Precision, Recall, F1-Score)', fontsize=13)
plt.legend(fontsize=12)
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure8_per_class_metrics.png', dpi=150)
plt.close()
print("Saved: figure8_per_class_metrics.png")

# ──────────────────────────────────────────────────────────────────────────────
# 13. SUMMARY 
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  NUMBERS TO COPY INTO YOUR REPORT")
print("=" * 55)
print(f"\nOverall Test Accuracy: {accuracy:.1f}%")
print(f"\nPer-Class Metrics:")
print(f"{'Category':12} {'Precision':12} {'Recall':10} {'F1-Score':10}")
print("-" * 46)
for i, cls in enumerate(classes):
    print(f"{cls:12} {precisions[i]:.2f}{'':8} {recalls[i]:.2f}{'':6} {f1s[i]:.2f}")

print(f"\nConfusion Matrix raw numbers:")
print(f"(Rows=Actual, Cols=Predicted, diagonal=correct)")
print(cm)
print(f"\nMean confidence on correct predictions  : {correct_confs.mean()*100:.1f}%")
print(f"Mean confidence on incorrect predictions: {incorrect_confs.mean()*100:.1f}%")

print("\n" + "=" * 55)
print("  FILES SAVED — insert these into your report:")
print("=" * 55)
print("  figure6_loss_curve.png         → Figure 6")
print("  figure7_confusion_matrix.png   → Figure 7 (Table 5 numbers above)")
print("  figure8_per_class_metrics.png  → Figure 8")
print("  figure11_confidence_distribution.png → Figure 11")
print("  classification_report.txt      → Table 4 numbers")
print("=" * 55)