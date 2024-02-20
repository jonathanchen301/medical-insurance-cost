import torch
import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(self, x)
        return output

    def learn(self, train_dataloader, dev_dataloader, num_epochs, optimizer, loss_fct):

        total_loss = 0.0
        correct = 0
        total = 0
        for epoch in range(num_epochs):
            print("Training Epoch " + str(epoch + 1) + "...")
            total_loss = 0.0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                predictions = self.forward(batch_X)
                loss = loss_fct(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_X.size(0)

                correct += (predictions == batch_y).sum().item()
                total += len(batch_y)
            average_loss = total_loss / len(train_dataloader)
            train_accuracy = correct / total

            correct = 0
            total = 0
            for batch_X, batch_y in dev_dataloader:
                predictions = self.predict(batch_X)
                correct += (predictions == batch_y).sum().item()
                total += len(batch_y)
            dev_accuracy = correct / total

            print(
                "Loss: "
                + str(average_loss)
                + " | Training Accuracy: "
                + str(train_accuracy)
                + " | Dev Accuracy: "
                + str(dev_accuracy)
            )
