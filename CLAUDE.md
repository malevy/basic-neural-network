# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BasicNeuralNet (BNN) is an educational neural network library built from scratch in C# to understand the underlying theory of neural networks. It implements forward propagation and backpropagation for fully-connected networks with various activation and loss functions.

**Important**: This is a learning/exploration project, not production-ready code. The math may contain errors and is not optimized for performance.

## Build and Test Commands

```bash
# Build the solution
dotnet build BasicNeuralNet.sln

# Run the main project (shows visualization using Plotly.NET)
dotnet run --project BNN/BNN.csproj

# Run all tests
dotnet test BNN.Tests/BNN.Tests.csproj

# Run a specific test class
dotnet test BNN.Tests/BNN.Tests.csproj --filter FullyQualifiedName~SigmoidActivationTests

# Run a specific test method
dotnet test BNN.Tests/BNN.Tests.csproj --filter FullyQualifiedName~SigmoidActivationTests.Derivative_ShouldMatchExpected
```

## Architecture

### Core Components

**Network** (`Network.cs`) - Top-level neural network class
- Composed of multiple `Layer` objects in sequence
- Takes aggregate and gradient loss functions as constructor parameters
- `Apply()`: Forward pass through all layers
- `Train()`: Forward pass + backpropagation with weight updates

**Layer** (`Layer.cs`) - Single layer in the network
- Contains an array of `Neuron` objects
- Associates neurons with an `IActivationFunction`
- `Apply()`: Computes neuron outputs, then applies activation function
- `BackProp()`: Calculates error gradients and propagates backwards to neurons

**Neuron** (`Neuron.cs`) - Individual computational unit
- Stores weights, bias, and momentum values
- `Apply()`: Weighted sum of inputs plus bias (no activation)
- `BackProp()`: Updates weights and bias using gradient descent with momentum

### Key Design Patterns

**Activation Function Caching**: Activation functions cache their inputs and outputs during the forward pass (`Squash()`) to reuse during backpropagation. This optimization avoids recalculating activation values when computing derivatives.

**Layer-wise Activation**: The `Neuron.Apply()` returns raw weighted sums. The `Layer` collects all neuron outputs and applies the activation function to the entire vector, which is critical for functions like Softmax that depend on all outputs.

**Gradient Flow**: During backpropagation, errors flow backward:
1. Loss function gradient → final layer activation function
2. Activation function derivative → neurons in that layer
3. Neurons calculate weight/bias updates and propagate error to previous layer
4. Repeat for each layer

### Activation Functions (`ActivationFunctions.cs`)

All implement `IActivationFunction`:
- `LinearFunction`: Identity (for regression)
- `ReLuFunction`: Rectified Linear Unit
- `LeakyReLuFunction`: ReLU variant with small negative slope
- `SigmoidFunction`: Binary classification (outputs 0-1)
- `TanhFunction`: Similar to sigmoid, outputs -1 to 1
- `SoftmaxFunction`: Multi-class classification (outputs sum to 1)

Each function defines custom weight initialization (e.g., He initialization for ReLU, Xavier/Glorot for others).

### Loss Functions (`LossFunctions.cs`)

Provides both aggregate error and gradient calculations:
- `SquaredError` / `SquaredErrorDerivative`: Regression problems
- `BinaryCrossEntropy` / `BinaryCrossEntropyDerivative`: Binary classification with sigmoid
- `CategoricalCrossEntropy` / `CategoricalCrossEntropyDerivative`: Multi-class with softmax

Note: `MeanError()` and `TotalError()` are helper wrappers that aggregate element-wise errors.

### NetworkBuilder (`NetworkBuilder.cs`)

Fluent API for constructing networks:
```csharp
var network = NetworkBuilder
    .WithInputs(2)
    .WithAggregateLossFunction(LossFunctions.MeanError(LossFunctions.SquaredError))
    .WithGradientLossFunction(LossFunctions.SquaredErrorDerivative)
    .WithLayer(4, new ActivationFunctions.ReLuFunction())
    .WithLayer(1, new ActivationFunctions.SigmoidFunction())
    .Build();
```

## Project Structure

- **BNN/** - Main library with core neural network implementation
  - `Tests/` - Example problems/scenarios (XOR, spiral, sine regression, Titanic dataset)
- **BNN.Tests/** - NUnit test suite for activation/loss functions
- **Target Framework**: .NET 6.0
- **Dependencies**: Plotly.NET for visualization

## Common Patterns

**Training Loop**: Examples in `BNN/Tests/*.cs` show typical patterns:
1. Build network with NetworkBuilder
2. Loop over epochs
3. For each training sample, call `network.Train(inputs, targets, learningRate)`
4. Monitor aggregate error, stop when threshold reached
5. Test with `network.Apply(inputs)`

**Activation/Loss Pairing**:
- Sigmoid + Binary Cross Entropy: Binary classification
- Softmax + Categorical Cross Entropy: Multi-class classification
- Linear + Mean Squared Error: Regression

**Single-layer limitation**: A single-layer network can only model linearly separable functions. Non-linear problems require at least 2 hidden layers with non-linear activation functions.

## Testing Notes

The test suite (`BNN.Tests/`) uses NUnit and tests individual activation functions and loss functions with known mathematical properties. The main project (`BNN/Tests/`) contains runnable examples that train networks on various problems.
