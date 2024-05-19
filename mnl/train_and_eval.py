import torch
from tqdm import tqdm
from .mnl_constants import (
    TRAINING_ITERATIONS,
    PATIENCE,
)

def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,   # TODO: should this be str?
    max_iter: int = TRAINING_ITERATIONS,
    patience: int = PATIENCE,
    verbose: bool = True,
) -> None:
    """
    Trains base model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_loss = float('inf')
    counter = 0
    if verbose:
        print(f"=> training on device {device}")
        print(f"=> max iterations: {TRAINING_ITERATIONS}")

    for epoch in tqdm(range(max_iter)):

        # training step
        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(data)    # logits
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"Epoch {epoch} => loss: {loss}")
    
        # validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                output = model(data)
                loss = criterion(output, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        if verbose:
            print(f"Epoch {epoch} => Validation loss: {val_loss}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    if verbose:
        print("Training complete.")



def test(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Tests base model.
    """
    model.eval()
    accuracy = 0.0

    with torch.no_grad():
        for n_batches, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            prediction = model(data)
            correct_prediction = torch.argmax(prediction, 1) == labels
            accuracy += correct_prediction.float().mean()

    accuracy = (100 * accuracy / n_batches)
    return accuracy