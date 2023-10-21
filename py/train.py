import wandb
import torch
from PIL import Image
import time


class Trainer:
    def __init__(
        self, model, train_loader, valid_loader, criterion, optimizer, device="cuda", project_name="tissue_ae"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.best_loss = float("inf")

        # Initialize wandb
        wandb.init(project=project_name)  # Set your project name
        wandb.watch(self.model)

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        for batch, (samples, labels) in enumerate(self.train_loader):
            print(
                f"Train | Epoch: {self.epoch}, Batch: {batch}/{len(self.train_loader)}",
                end="\r",
            )
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(samples)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * samples.size(0)
        
        wandb.log({"train_loss": train_loss / len(self.train_loader.dataset)})

        self.epoch += 1

        return train_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch, (samples, labels) in enumerate(self.valid_loader):
                print(f"Valid | Batch: {batch}/{len(self.valid_loader)}", end="\r")
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item() * samples.size(0)
        print(f"Valid Loss: {valid_loss / len(self.valid_loader.dataset)}")
        return valid_loss / len(self.valid_loader.dataset)

    def run(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            valid_loss = self.validate()

            # Check if this epoch has the best validation loss and save model
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                time_formated = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                torch.save(self.model.state_dict(), f"best_model_predictor.pt")

            # Log metrics to wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

        # Close the wandb run
        wandb.finish()

    def show_reconstruction(self, dataloader):
        """Show a single sample and its reconstruction from the model"""
        self.model.eval()
        with torch.no_grad():
            for batch, samples in enumerate(dataloader):
                samples = samples.to(self.device)
                outputs = self.model(samples)
                # Convert the tensors to numpy arrays
                samples = samples.cpu().numpy()
                outputs = outputs.cpu().numpy()

                # PIL images
                sample = Image.fromarray(samples[0][0] * 255).convert("RGB")
                output = Image.fromarray(outputs[0][0] * 255).convert("RGB")
                sample.show()
                output.show()
                break

    def save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
