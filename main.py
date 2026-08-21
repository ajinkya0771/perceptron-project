import numpy as np

# 1. TRAINING DATA

# X contains the input features.
#
# We are creating an AND gate:
#
# Input        Expected Output
# [0, 0]   ->       0
# [0, 1]   ->       0
# [1, 0]   ->       0
# [1, 1]   ->       1
#
# Each row represents one training example.
# Each row has 2 input features: x1 and x2.

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# y contains the correct/expected output for each input.
#
# y[0] corresponds to X[0] -> [0, 0] -> 0
# y[1] corresponds to X[1] -> [0, 1] -> 0
# y[2] corresponds to X[2] -> [1, 0] -> 0
# y[3] corresponds to X[3] -> [1, 1] -> 1

y = np.array([0, 0, 0, 1])


# 2. INITIALIZE WEIGHTS AND BIAS

# A perceptron needs weights to determine the importance
# of each input feature.
#
# Since X has 2 features, we need 2 weights:
#
#     x1 -> weight w1
#     x2 -> weight w2
#
# X.shape gives the dimensions of X:
# X.shape = (4, 2)
#
# X.shape[1] = 2
#
# Therefore:
# weights = [0, 0]

# We start with zero weights.

weights = np.zeros(X.shape[1])


# Bias is an additional value that helps the perceptron
# shift the decision boundary.
#
# Initially, we set bias to 0.

bias = 0.0


# 3. HYPERPARAMETERS

# Learning rate controls how much the weights and bias
# change after every incorrect prediction.
#
# Formula:
#
# New Weight = Old Weight + Learning Rate * Error * Input
#
# A larger learning rate -> larger updates
# A smaller learning rate -> smaller updates

learning_rate = 0.1


# Epoch means one complete pass through the entire
# training dataset.
#
# Here we allow the perceptron to see the complete dataset
# up to 10 times.

epochs = 10


# 4. TRAINING THE PERCEPTRON

# range(epochs) generates:
#
# 0, 1, 2, ..., 9
#
# So the training loop can run for a maximum of 10 epochs.

for epoch in range(epochs):

    print("\n" + "-" * 50)
    print(f"Epoch {epoch + 1}")
    print("-" * 50)


    # Display the weights and bias at the beginning
    # of the current epoch.
    #
    # These values will change during training.

    print(f"\nInitial Weights: {weights}")
    print(f"Initial Bias: {bias}")


    # This variable counts how many incorrect predictions
    # the perceptron makes during the current epoch.
    #
    # We start with zero errors.

    errors = 0


    # 5. PROCESS EACH TRAINING EXAMPLE

    # len(X) = 4 because we have 4 training examples.
    #
    # i will take the values:
    # 0, 1, 2, 3

    for i in range(len(X)):


        # Select the current input row.
        #
        # Example:
        # If i = 0:
        # inputs = [0, 0]
        #
        # If i = 3:
        # inputs = [1, 1]

        inputs = X[i]


        # Get the correct/expected output for
        # the current input.

        actual_outputs = y[i]


        # 6. CALCULATE WEIGHTED SUM

        # The perceptron first calculates:
        #
        # weighted_sum = x1*w1 + x2*w2 + bias
        #
        # This can also be written mathematically as:
        #
        # z = X.W + b
        #
        # np.dot(inputs, weights) calculates:
        #
        # x1*w1 + x2*w2
        #
        # Then we add the bias.

        weighted_sum = np.dot(inputs, weights) + bias


        # 7. MAKE A PREDICTION

        # The perceptron uses a STEP ACTIVATION FUNCTION.
        #
        # If weighted_sum >= 0:
        #       prediction = 1
        #
        # Otherwise:
        #       prediction = 0
        #
        # This converts the numerical weighted sum
        # into a binary prediction.

        if weighted_sum >= 0:
            prediction = 1
        else:
            prediction = 0


        # 8. CALCULATE ERROR

        # Compare the actual/expected output with
        # the prediction.
        #
        # Formula:
        #
        # Error = Actual Output - Prediction
        #
        # Example:
        #
        # Actual = 1
        # Prediction = 0
        # Error = 1 - 0 = +1
        #
        # Actual = 0
        # Prediction = 1
        # Error = 0 - 1 = -1
        #
        # Actual = 0
        # Prediction = 0
        # Error = 0 - 0 = 0
        #
        # Error = 0 means the prediction was correct.

        loss = actual_outputs - prediction


        # If the error is not zero, the prediction was incorrect.
        #
        # Increase the error counter by 1.

        if loss != 0:
            errors += 1


        # 9. UPDATE WEIGHTS

        # The perceptron learns by changing its weights
        # whenever it makes a wrong prediction.
        #
        # Formula:
        #
        # New Weight =
        # Old Weight + Learning Rate * Error * Input
        #
        # In Python:
        #
        # weights = weights + learning_rate * loss * inputs
        #
        # Notice that NumPy performs this calculation
        # element by element for both weights.

        weights = weights + learning_rate * loss * inputs


        # 10. UPDATE BIAS

        # The bias is updated using:
        #
        # New Bias =
        # Old Bias + Learning Rate * Error
        #
        # The bias does not multiply by the input because
        # the bias acts like an additional constant input of 1.

        bias = bias + learning_rate * loss


        # 11. DISPLAY TRAINING DETAILS

        # Print the current training example so that
        # we can understand how the perceptron is learning.

        print(f"\nInput: {inputs}")

        # Print the correct answer.

        print(f"Actual output: {actual_outputs}")

        # Print what the perceptron predicted.

        print(f"Prediction: {prediction}")

        # Print the error made by the perceptron.

        print(f"Error: {loss}")

        # Print the new weights after the update.

        print(f"Updated Weights: {weights}")

        # Print the new bias after the update.
        #
        # :.2f means display the number with 2 decimal places.

        print(f"Updated Bias: {bias:.2f}")


    # 12. CHECK ERRORS FOR THE CURRENT EPOCH

    # After processing all 4 training examples,
    # display how many incorrect predictions occurred.

    print(f"\nErrors in epoch: {errors}")


    # 13. EARLY STOPPING

    # If errors == 0, it means the perceptron correctly
    # classified every training example in this epoch.
    #
    # Therefore, there is no need to continue training.

    if errors == 0:

        print("\nTraining is successful")

        # break immediately stops the training loop.

        break


# 14. TEST THE TRAINED PERCEPTRON


# At this point, training is finished.
#
# The perceptron should now have learned suitable values
# for the weights and bias.

print("\n")
print("-" * 50)
print("FINAL MODEL")
print("-" * 50)


# Display the final learned weights.

print(f"Weights: {weights}")


# Display the final learned bias.

print(f"Bias: {bias:.2f}")


# 15. MAKE PREDICTIONS USING THE TRAINED MODEL


# Go through every input example again.

for i in range(len(X)):

    # Get the input.

    inputs = X[i]


    # Get the expected answer.
    #
    # We use this only for comparison during testing.

    actual_outputs = y[i]


    # Calculate the weighted sum using the FINAL weights
    # and FINAL bias learned during training.
    #
    # Formula:
    #
    # z = x1*w1 + x2*w2 + bias

    weighted_sum = np.dot(inputs, weights) + bias


    # Apply the step activation function again.
    #
    # weighted_sum >= 0 -> 1
    # weighted_sum < 0  -> 0

    if weighted_sum >= 0:
        prediction = 1
    else:
        prediction = 0


    # Display the final result.
    #
    # Input     -> what we gave to the model
    # Expected  -> correct answer
    # Predicted -> answer produced by the perceptron


    print(
        f"Input: {inputs}",
        f"Expected: {actual_outputs}",
        f"Predicted: {prediction}"
    )