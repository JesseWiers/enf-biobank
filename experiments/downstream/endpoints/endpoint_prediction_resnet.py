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
import time  # Add this import

# Change imports to use the new dataset
from experiments.datasets.biobank_dataset_endpoint_images import ImageEndpointDataset, collate_fn_images_endpoint
from experiments.downstream_models.ResNet import ResNet50
import matplotlib.pyplot as plt
import sklearn.metrics

def create_dataloaders(nifti_root_path, endpoint_name, batch_size=32, 
                      train_split=0.75, val_split=0.15,  # Modified split ratios
                      random_seed=42, num_workers=4, debug_limit=None,
                      z_indices=3, t_indices=-1):
    """
    Create train, validation, and test dataloaders with a 75/15/15 split.
    """
    np.random.seed(random_seed)
    
    dataset = ImageEndpointDataset(
        nifti_root=nifti_root_path,
        endpoint_name=endpoint_name,
        z_indices=z_indices,
        t_indices=t_indices,
        debug_limit=debug_limit,
        random_seed=random_seed
    )
    
    # Separate indices by label
    case_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
    control_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    
    # Calculate split sizes for each group
    n_train_cases = int(len(case_indices) * train_split)
    n_val_cases = int(len(case_indices) * val_split)
    # Test cases will be the remainder
    
    n_train_controls = int(len(control_indices) * train_split)
    n_val_controls = int(len(control_indices) * val_split)
    # Test controls will be the remainder
    
    # Shuffle indices
    np.random.shuffle(case_indices)
    np.random.shuffle(control_indices)
    
    # Split each group into train/val/test
    train_indices = (
        case_indices[:n_train_cases] +  # Training cases
        control_indices[:n_train_controls]  # Training controls
    )
    
    val_indices = (
        case_indices[n_train_cases:n_train_cases + n_val_cases] +  # Validation cases
        control_indices[n_train_controls:n_train_controls + n_val_controls]  # Validation controls
    )
    
    test_indices = (
        case_indices[n_train_cases + n_val_cases:] +  # Test cases
        control_indices[n_train_controls + n_val_controls:]  # Test controls
    )
    
    # Shuffle the combined indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        collate_fn=collate_fn_images_endpoint
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        collate_fn=collate_fn_images_endpoint
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers=num_workers,
        collate_fn=collate_fn_images_endpoint
    )
    
    return train_loader, val_loader, test_loader

def get_config():
    config = ml_collections.ConfigDict()
    
    # Fixed seed for dataset splitting - this will always be the same
    config.dataset_seed = 42
    # Separate seed for model initialization - change this for different model initializations
    config.model_seed = 42  # You can change this value to try different model initializations

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 4
    config.dataset.nifti_root = '/projects/prjs1252/data_jesse_v2/nifti_dataset_cropped/'
    config.dataset.endpoint_name = 'cardiomyopathy'
    config.dataset.debug_limit = None
    config.dataset.z_indices = 4  # uses middle 4 z indices
    config.dataset.t_indices = (0, 10, 20, 30, 40, 49)  # specific timepoints

    # Model config
    config.model = ml_collections.ConfigDict()
    config.model.name = "ResNet50"
    config.model.num_classes = 2
    config.model.in_channels = None  # Will be set based on data

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 32
    config.train.num_epochs = 100
    config.train.learning_rate = 0.001
    
    # Experiment naming
    config.exp_name = "endpoint_prediction_resnet"
    config.run_name = "baseline"
    
    return config

_CONFIG = config_flags.DEFINE_config_dict("config", get_config())

def main(_):
    config = _CONFIG.value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    torch.manual_seed(config.model_seed)  # Changed from config.seed
    np.random.seed(config.model_seed)     # Changed from config.seed
    
    # Initialize wandb
    run = wandb.init(
        project=config.exp_name,
        config=config.to_dict(),
        name=config.run_name
    )
    
    # Create dataloaders with dataset seed
    train_loader, val_loader, test_loader = create_dataloaders(
        nifti_root_path=config.dataset.nifti_root,
        endpoint_name=config.dataset.endpoint_name,
        batch_size=config.train.batch_size,
        train_split=0.75,
        val_split=0.15,
        num_workers=config.dataset.num_workers,
        debug_limit=config.dataset.debug_limit,
        random_seed=config.dataset_seed,  # Use dataset_seed for consistent splits
        z_indices=config.dataset.z_indices,
        t_indices=config.dataset.t_indices
    )
    
    # Calculate total input channels based on config
    if isinstance(config.dataset.t_indices, tuple):
        num_timepoints = len(config.dataset.t_indices)
    else:
        num_timepoints = 1
        
    if isinstance(config.dataset.z_indices, int):
        num_z_slices = 1
    else:
        num_z_slices = len(config.dataset.z_indices)
    
    total_channels = num_timepoints * config.dataset.z_indices

    # Create model with dynamic number of input channels
    model = ResNet50(
        num_classes=2,  # Binary classification
        channels=total_channels  # Dynamic based on T and Z config
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    global_step = 0
    
    for epoch in range(config.train.num_epochs):
        # Training phase
        train_start_time = time.time()
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch_idx, (patient_ids, images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            # Convert numpy arrays to torch tensors
            images = torch.from_numpy(images).float()
            labels = torch.from_numpy(labels).long()
            
            # Reshape images
            B, H, W, T, _, Z = images.shape
            images = images.permute(0, 3, 4, 5, 1, 2)  # [B, T, 1, Z, H, W]
            images = images.reshape(B, T * Z, H, W)    # [B, C, H, W]
            
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
                
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Add accuracy tracking
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            global_step += 1  # Increment after each batch
            
        train_duration = time.time() - train_start_time  # Move this here, right after training loop
        
        # Now we can calculate accuracy
        train_acc = train_correct / train_total
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        val_start_time = time.time()
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        misclassified_data = []
        
        with torch.no_grad():
            for patient_ids, images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                
                images = torch.from_numpy(images).float()
                labels = torch.from_numpy(labels).long()
            
                B, H, W, T, _, Z = images.shape

                images = images.permute(0, 3, 4, 5, 1, 2)  # [B, T, 1, Z, H, W]
                images = images.reshape(B, T * Z, H, W) 
                
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
                        'predicted': "Case" if predicted[idx].item() == 1 else "Control",
                        'actual': "Case" if labels[idx].item() == 1 else "Control"
                    })
        
        val_duration = time.time() - val_start_time
        val_acc = val_correct / val_total
        avg_val_loss = np.mean(val_losses)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()  # Make sure to copy the state dict
            logging.info(f"New best validation accuracy: {val_acc:.4f}")
        
        # Log metrics with global_step and timing
        wandb.log({
            "train/loss": avg_train_loss,
            "train/accuracy": train_acc,
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
            "epoch": epoch,
        }, step=epoch)
        
        # Log misclassifications
        if misclassified_data:
            misclassified_table = wandb.Table(
                columns=["patient_id", "predicted", "actual"]
            )
            for item in misclassified_data:
                # When adding data to the misclassified table, convert patient_id to str
                misclassified_table.add_data(
                    str(item['patient_id']),  # Convert numpy string to regular Python string
                    "Case" if item['predicted'] == 1 else "Control",
                    "Case" if item['actual'] == 1 else "Control"
                )
            wandb.log({f"val_errors/misclassified_patients_epoch_{epoch}": misclassified_table})
        
        # Log to console with timing
        logging.info(f"Epoch {epoch}:")
        logging.info(f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}, Time: {train_duration:.2f}s")
        logging.info(f"  Val   - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Load best model for final evaluation
    logging.info(f"Loading best model (validation accuracy: {best_val_acc:.4f}) for test evaluation")
    model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    logging.info("Performing final evaluation on test set...")
    test_start_time = time.time()
    model.eval()
    test_losses = []
    test_correct = 0
    test_total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for patient_ids, images, labels in tqdm(test_loader, desc="Final test evaluation"):
            images = torch.from_numpy(images).float()
            labels = torch.from_numpy(labels).long()
            
            B, H, W, T, _, Z = images.shape
            images = images.permute(0, 3, 4, 5, 1, 2)
            images = images.reshape(B, T * Z, H, W)
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Store for AUROC calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    
    test_duration = time.time() - test_start_time
    test_acc = test_correct / test_total
    avg_test_loss = np.mean(test_losses)
    
    # Calculate AUROC
    auroc = sklearn.metrics.roc_auc_score(np.array(all_labels), np.array(all_predictions))
    
    # Log final test results
    logging.info("Final Test Results (using best validation model):")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logging.info(f"Test Loss: {avg_test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test AUROC: {auroc:.4f}")
    
    # Log to wandb
    wandb.log({
        "test/loss": avg_test_loss,
        "test/accuracy": test_acc,
        "test/auroc": auroc,
        "test/num_samples": test_total,
        "best_validation_accuracy": best_val_acc
    })
    
    # Plot and save ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(np.array(all_labels), np.array(all_predictions))
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    wandb.log({"test/roc_curve": wandb.Image(plt)})
    plt.close()

    run.finish()

if __name__ == "__main__":
    app.run(main)