import ml_collections
from ml_collections import config_flags
from absl import app
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
import wandb
import numpy as np

from experiments.datasets.biobank_image_dataset import create_dataloaders
from ResNet.ResNet import ResNet50

def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 68

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 4
    config.dataset.nifti_root = "/projects/prjs1252/data_jesse/cmr_cropped"
    config.dataset.csv_path = "/projects/prjs1252/data_jesse/metadata/filtered_endpoints.csv"
    config.dataset.debug_limit = None

    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.name = "ResNet50"
    config.model.num_classes = 2
    config.model.in_channels = 2  # ED and ES frames

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 32
    config.train.num_epochs = 100
    config.train.validation_interval = 1
    config.train.learning_rate = 0.001
    
    # Experiment naming
    config.exp_name = "lvef_prediction_resnet"
    config.run_name = "baseline"
    
    return config

_CONFIG = config_flags.DEFINE_config_dict("config", get_config())

def main(_):
    config = _CONFIG.value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize wandb
    run = wandb.init(
        project=config.exp_name,
        config=config.to_dict(),
        name=config.run_name
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        nifti_root_path=config.dataset.nifti_root,
        endpoints_csv_path=config.dataset.csv_path,
        batch_size=config.train.batch_size,
        num_workers=config.dataset.num_workers,
        debug_limit=config.dataset.debug_limit
    )
    
    # Create model
    model = ResNet50(
        num_classes=config.model.num_classes,
        channels=config.model.in_channels
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(config.train.num_epochs):
        model.train()
        train_losses = []
        
        for batch_idx, (patient_ids, images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
        # Validation phase
        if (epoch + 1) % config.train.validation_interval == 0:
            model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for patient_ids, images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                    
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
            
            # Log metrics
            wandb.log({
                "train/loss": np.mean(train_losses),
                "val/loss": np.mean(val_losses),
                "val/accuracy": val_acc,
                "val/best_accuracy": best_val_acc
            }, step=epoch)
    
    # Test phase using best model and rotated images
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_correct = 0
    test_total = 0
    misclassified_data = []
    
    with torch.no_grad():
        for patient_ids, images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Track misclassifications
            misclassified_mask = predicted.ne(labels)
            for idx in torch.where(misclassified_mask)[0]:
                misclassified_data.append({
                    'patient_id': patient_ids[idx],
                    'predicted': "LVEF ≥ 40" if predicted[idx].item() == 1 else "LVEF < 40",
                    'actual': "LVEF ≥ 40" if labels[idx].item() == 1 else "LVEF < 40"
                })
    
    test_acc = test_correct / test_total
    
    # Log final test results
    wandb.log({
        "test/accuracy": test_acc,
        "test/num_misclassified": len(misclassified_data)
    })
    
    if misclassified_data:
        misclassified_table = wandb.Table(
            columns=["patient_id", "predicted", "actual"]
        )
        for item in misclassified_data:
            misclassified_table.add_data(
                item['patient_id'],
                item['predicted'],
                item['actual']
            )
        wandb.log({"test/misclassified_patients": misclassified_table})
    
    run.finish()

if __name__ == "__main__":
    app.run(main)