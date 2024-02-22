import torch
import torch.nn as nn
import wandb
import csv


class LinearRegression(nn.Module):

    def __init__(self, feature_size):
        super().__init__()
        self.linear = nn.Linear(feature_size, 1)

    def forward(self, x):
        return self.linear.forward(x)

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

            wandb.log({"train_loss": train_loss, "dev_loss": dev_loss})

            print("Train Loss: " + str(train_loss) + " | Dev Loss: " + str(dev_loss))

    def evaluate(self, test_dataloader, loss_fct, output_csv=None):

        print("Evaluating Model On Test Set")

        total_test_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch_X, batch_y in test_dataloader:
            predictions = self.predict(batch_X)
            loss = loss_fct(predictions, batch_y)
            total_test_loss += loss.item() * batch_X.size(0)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch_y.tolist())
        test_loss = total_test_loss / len(test_dataloader.dataset)

        print("Test Loss: " + str(test_loss))

        if output_csv:
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Prediction", "Actual"])
                for pred, actual in zip(all_predictions, all_labels):
                    writer.writerow([pred[0], actual])


class MultilayerPerceptron(nn.Module):

    def __init__(self, feature_size, hidden_dims):
        super().__init__()

        self.linear1 = nn.Linear(feature_size, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dims, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

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

            wandb.log({"train_loss": train_loss, "dev_loss": dev_loss})

            print("Train Loss: " + str(train_loss) + " | Dev Loss: " + str(dev_loss))

    def evaluate(self, test_dataloader, loss_fct, output_csv=None):

        print("Evaluating Model On Test Set")

        total_test_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch_X, batch_y in test_dataloader:
            predictions = self.predict(batch_X)
            loss = loss_fct(predictions, batch_y)
            total_test_loss += loss.item() * batch_X.size(0)

            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch_y.tolist())
        test_loss = total_test_loss / len(test_dataloader.dataset)

        print("Test Loss: " + str(test_loss))

        if output_csv:
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Prediction", "Actual"])
                for pred, actual in zip(all_predictions, all_labels):
                    writer.writerow([pred[0], actual])
