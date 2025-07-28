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

# Change the import to use the new dataset
from experiments.datasets.biobank_dataset_lvef_images import BiobankNiftiEDESMiddleSlices, collate_fn_images
from experiments.downstream_models.ResNet import ResNet50

def create_dataloaders(nifti_root_path, endpoints_csv_path, batch_size=32, 
                      val_split=0.2, random_seed=42, num_workers=4, debug_limit=None):
    """
    Create train, validation, and test dataloaders.
    """
    # Create dataset
    full_dataset = BiobankNiftiEDESMiddleSlices(
        root=nifti_root_path,
        endpoints_csv_path=endpoints_csv_path,
        debug_limit=debug_limit
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    test_size = val_size
    train_size = total_size - (val_size + test_size)
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_images
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_images
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_images
    )
    
    return train_loader, val_loader, test_loader

def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 68

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 4
    config.dataset.nifti_root = '/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped/'
    config.dataset.csv_path = '/projects/prjs1252/data_jesse_v2/metadata/filtered_endpoints.csv'
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
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (patient_ids, images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            # Modify how we handle the images since we get (ed, es) tuple
            images_ed, images_es = images
            # Stack ED and ES along channel dimension
            images = torch.cat([images_ed, images_es], dim=3)  # [B, H, W, 2]
            # Permute to channel-first format for ResNet
            images = images.permute(0, 3, 1, 2)  # [B, 2, H, W]
            
            # Convert LVEF values to binary labels
            labels = (labels >= 40.0).long()
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Calculate and log average training loss for the epoch
        avg_train_loss = np.mean(train_losses)
        logging.info(f"Epoch {epoch}: train-Loss = {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        misclassified_data = []  # Track misclassifications during validation
        
        with torch.no_grad():
            for patient_ids, images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                images_ed, images_es = images
                images = torch.cat([images_ed, images_es], dim=3)
                images = images.permute(0, 3, 1, 2)
                labels = (labels >= 40.0).long()
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Track misclassifications
                misclassified_mask = predicted.ne(labels)
                for idx in torch.where(misclassified_mask)[0]:
                    misclassified_data.append({
                        'patient_id': patient_ids[idx],
                        'predicted': "LVEF ≥ 40" if predicted[idx].item() == 1 else "LVEF < 40",
                        'actual': "LVEF ≥ 40" if labels[idx].item() == 1 else "LVEF < 40"
                    })
        
        # Calculate validation metrics
        avg_val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        # Log metrics
        wandb.log({
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
            "val/best_accuracy": best_val_acc,
            "epoch": epoch
        })
        
        # Log validation misclassifications
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
            wandb.log({f"val_errors/misclassified_patients_epoch_{epoch}": misclassified_table})
        
        # Log to console
        logging.info(f"Epoch {epoch}: val-Loss = {avg_val_loss:.4f}, val-Accuracy = {val_acc:.4f}")
        if misclassified_data:
            logging.info(f"\nMisclassified patients in epoch {epoch}:")
            for item in misclassified_data[:5]:  # Show first 5 misclassifications
                logging.info(
                    f"Patient {item['patient_id']}: "
                    f"Predicted = {item['predicted']}, "
                    f"Actual = {item['actual']}"
                )

    # Test phase
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_correct = 0
    test_total = 0
    misclassified_data = []
    
    with torch.no_grad():
        for patient_ids, images, labels in tqdm(test_loader, desc="Testing"):
            images_ed, images_es = images
            images = torch.cat([images_ed, images_es], dim=3)
            images = images.permute(0, 3, 1, 2)
            labels = (labels >= 40.0).long()
            
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