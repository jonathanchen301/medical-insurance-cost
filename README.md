# medical-insurance-cost

## Purpose

The goal of this project is to learn to implement a linear regression and a multi-layer perceptron using PyTorch
to predict insurance charges (individual medical costs billed by health insurance) based on individual age, sex,
bmi, whether they have chidlren, if they are a smoker, and what region they live in the United States.

Although the subgoal is to predict charges accurately based on these features, the main goal of this project is to
familiarize myself with the PyTorch library and how to implement a neural network using a simple dataset.

## Approach

First, I used PyTorch's DataLoader to load in the dataset. The DataLoader performs categorical encoding which transforms
categorical data to integer data. I then separated the data using a 80-10-10 split. First, I implemented a linear regression
using PyTorch, and then a multilayer perceptron (MLP). For the MLP, I tuned the hyperarameters of learning rate, number of epochs,
number of hidden layers, number of dimensions within each hidden layer. I also changed from stochastic gradient descent to ADAM
optimizer which yielded better results.

## Language + Libraries Used

Python | Libraries: pandas, torch, sklearn

## Credit

Dataset obtained from [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Results

Check predictions.csv -- it shows the predicted value of my final MLP model and the actual value. Most of the predictions are
reasonable. For example, prediction 5 yielded 13617.65234375 while the actual cost was 12644.5888671875. Still, some predictions were
highly inaccurate such as prediction 36 which yielded 11195.5888671875 while the actual cost was 23045.56640625.

However, upon examining the data point that yielded this prediction:

52,female,30.875,0,no,northeast,23045.56616

and comparing it to another data point:

55,female,29.83,0,no,northeast,11286.5387

we observe that the second data point is 3 years older, the same gender, lower bmi, no children, from the same area, but had an insurance
cost that was half of the data point. From this we can see that the features provided by the dataset do not capture all the
important factors that contribute to the insurance cost. Some other factors include medical history, family history, other lifestyle
factors aside from smoking, type of insurance plan, family size aside from children, economic factors such as inflation.

Without these other features, we can see that similar data points have very different medical costs, so certain outliers were not accurately predicted using my model.

## Next Steps

Create a full-stack website that gives users a initial estimate on their medical insurance costs based only on a few factors: age, sex, bmi, # of children, if they are a smoker or not,
and region of residence.
