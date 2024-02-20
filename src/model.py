import torch
import torch.nn as nn
import wandb


class LinearRegression(nn.Module):

    def __init__(self, feature_size):
        super().__init__()
        self.linear = nn.Linear(feature_size, 1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return output

    def learn(self, train_dataloader, dev_dataloader, num_epochs, optimizer, loss_fct):

        for epoch in range(num_epochs):
            print("Training Epoch " + str(epoch + 1) + "...")
            total_train_loss = 0.0
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                predictions = self.forward(batch_X)
                loss = loss_fct(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch_X.size(0)
            train_loss = total_train_loss / len(train_dataloader.dataset)

            total_dev_loss = 0.0
            for batch_X, batch_y in dev_dataloader:
                predictions = self.predict(batch_X)
                loss = loss_fct(predictions, batch_y)
                total_dev_loss += loss.item() * batch_X.size(0)
            dev_loss = total_dev_loss / len(dev_dataloader.dataset)

            # wandb.log({"train_loss": train_loss, "dev_loss": dev_loss})

            print("Train Loss: " + str(train_loss) + " | Dev Loss: " + str(dev_loss))

    def evaluate(self, test_dataloader, loss_fct):

        print("Evaluating Model On Test Set")

        total_test_loss = 0.0
        for batch_X, batch_y in test_dataloader:
            predictions = self.predict(batch_X)
            loss = loss_fct(predictions, batch_y)
            total_test_loss += loss.item() * batch_X.size(0)
        test_loss = total_test_loss / len(test_dataloader.dataset)

        print("Test Loss: " + str(test_loss))

        return test_loss
