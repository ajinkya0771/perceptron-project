# Perceptron from Scratch using NumPy

A simple implementation of the **Perceptron Learning Algorithm from scratch using Python and NumPy**.

This project demonstrates how a single-layer perceptron learns to classify the **AND logical operation** using weights, bias, a learning rate, and iterative weight updates.

The perceptron learning algorithm is implemented manually without using machine learning libraries such as Scikit-learn.

---

## 🎯 Objective

The objective of this project is to train a perceptron to learn the **AND gate**.

### AND Truth Table

| Input X₁ | Input X₂ | Expected Output |
|----------|----------|-----------------|
| 0        | 0        | 0               |
| 0        | 1        | 0               |
| 1        | 0        | 0               |
| 1        | 1        | 1               |

---

## 🧠 How a Perceptron Works

A perceptron takes input features, multiplies them by their corresponding weights, adds a bias, and produces a prediction using a step activation function.

### 1. Weighted Sum

~~~
Z = X₁ × W₁ + X₂ × W₂ + Bias
~~~

In Python:

~~~python
weighted_sum = np.dot(inputs, weights) + bias
~~~

### 2. Step Activation Function

~~~
If Z >= 0:
    Prediction = 1
Else:
    Prediction = 0
~~~

### 3. Error Calculation

~~~
Error = Actual Output - Prediction
~~~

### 4. Weight Update

~~~
New Weight = Old Weight + Learning Rate × Error × Input
~~~

In Python:

~~~python
weights = weights + learning_rate * loss * inputs
~~~

### 5. Bias Update

~~~
New Bias = Old Bias + Learning Rate × Error
~~~

In Python:

~~~python
bias = bias + learning_rate * loss
~~~

---

## 🔄 Training Process

The perceptron follows this learning process:

~~~
Input Data
    ↓
Calculate Weighted Sum
    ↓
Apply Step Function
    ↓
Generate Prediction
    ↓
Calculate Error
    ↓
Update Weights & Bias
    ↓
Better Prediction
    ↓
Repeat
~~~

---

## 🛠️ Technologies Used

- Python
- NumPy
- uv
- Python Virtual Environment

---

## 📁 Project Structure

~~~
Perceptron/
│
├── main.py
├── requirements.txt
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
~~~

### File Description

| File | Description |
|------|-------------|
| `main.py` | Complete perceptron implementation |
| `requirements.txt` | Project dependencies |
| `README.md` | Project documentation |
| `pyproject.toml` | Python project configuration |
| `uv.lock` | Locked dependency versions |
| `.gitignore` | Files and folders excluded from Git |

---

## 📦 Installation

### 1. Clone the Repository

~~~bash
git clone <your-github-repository-url>
cd Perceptron
~~~

### 2. Create a Virtual Environment

~~~bash
uv venv
~~~

### 3. Activate the Virtual Environment

For Windows:

~~~bash
.venv\Scripts\activate
~~~

### 4. Install Dependencies

~~~bash
uv pip install -r requirements.txt
~~~

### 5. Verify Installation

~~~bash
uv pip list
~~~

---

## ▶️ Run the Project

Run the perceptron using:

~~~bash
uv run python main.py
~~~

Or, after activating the virtual environment:

~~~bash
python main.py
~~~

---

## ⚙️ Hyperparameters

The model uses the following hyperparameters:

~~~python
learning_rate = 0.1
epochs = 10
~~~

### Learning Rate

The learning rate controls how much the weights and bias change after an incorrect prediction.

~~~
Learning Rate = 0.1
~~~

A larger learning rate produces larger updates, while a smaller learning rate produces smaller updates.

### Epochs

An epoch represents one complete pass through the training dataset.

~~~
Epochs = 10
~~~

The program stops early if the model achieves zero classification errors during an epoch.

---

## 📊 Training Result

The perceptron successfully learns the AND gate and converges in **4 epochs**.

Final learned parameters:

~~~
Weights: [0.2 0.1]
Bias: -0.20
~~~

### Final Predictions

| Input | Expected | Predicted |
|-------|----------|-----------|
| `[0 0]` | 0 | 0 |
| `[0 1]` | 0 | 0 |
| `[1 0]` | 0 | 0 |
| `[1 1]` | 1 | 1 |

The trained model correctly classifies all four examples.

~~~
Training Accuracy = 100%
~~~

---

## 🔍 Example Calculation

Using the final model:

~~~
Weights = [0.2, 0.1]
Bias = -0.2
~~~

For input:

~~~
[1, 1]
~~~

The weighted sum is:

~~~
Z = (1 × 0.2) + (1 × 0.1) - 0.2

Z = 0.1
~~~

Since:

~~~
0.1 >= 0
~~~

the step function produces:

~~~
Prediction = 1
~~~

This matches the expected AND gate output.

---

## 📚 Concepts Demonstrated

This project demonstrates:

- Perceptron
- Artificial neuron
- Input features
- Weights
- Bias
- Weighted sum
- Step activation function
- Binary classification
- Error calculation
- Learning rate
- Weight updates
- Bias updates
- Epochs
- Model convergence
- Training and testing

---

## ⚠️ Limitations

A single perceptron can only solve **linearly separable problems**.

### Can Solve

~~~
AND
OR
~~~

### Cannot Solve

~~~
XOR
~~~

The XOR problem is not linearly separable and requires a more complex neural network with additional layers.

---

## 🚀 Future Improvements

- Implement the OR gate
- Demonstrate the XOR problem
- Visualize the decision boundary
- Add accuracy calculation
- Add confusion matrix
- Experiment with different learning rates
- Experiment with different numbers of epochs
- Implement a Multi-Layer Perceptron
- Compare the implementation with Scikit-learn's Perceptron

---

## 💡 Key Learning

The main idea behind the perceptron is that the model **learns by adjusting its weights and bias based on prediction errors**.

~~~
Prediction
    ↓
Calculate Error
    ↓
Update Weights & Bias
    ↓
Better Prediction
    ↓
Repeat
~~~

Once the model correctly classifies the training examples, the training process can stop.

---

## 👨‍💻 Author

**Ajinkya Dhote**

B.Tech – Artificial Intelligence & Machine Learning

---

## ⭐ Conclusion

This project provides a simple **Perceptron implementation from scratch using NumPy** and demonstrates the fundamental learning mechanism of a basic artificial neuron.

The trained perceptron successfully learns the **AND logical operation** and correctly classifies all four input combinations with **100% accuracy**.