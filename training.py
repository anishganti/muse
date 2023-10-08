from muse import BaseTransformer,BaseTransformerConfig
import torch

class Trainer(nn.Module):
    def __init__():
    # Define hyperparameters
        batch_size = 64
        num_epochs = 10
        learning_rate = 0.001

        # Initialize transformer model
        model = BaseTransformer(BaseTransformerConfig())

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # TODO: load dataset properly 

        # Move the model to the appropriate device (CPU or GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    def train(self):
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute the loss
                loss = self.criterion(outputs.view(-1, output_dim), targets.view(-1))  # Modify as needed

                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}/{len(self.dataloader)}], Loss: {total_loss / (batch_idx + 1):.4f}')

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {total_loss / len(self.dataloader):.4f}')

        # Save the trained model
        torch.save(self.model.state_dict(), 'path')
