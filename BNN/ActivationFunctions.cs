using System;
using System.Diagnostics;
using System.Linq;

namespace BNN;

public interface IActivationFunction
{
    double[] Squash(double[] inputs);
    double[] BackProp(double[] errorWrtOutput);
    double[] WeightInitializers(int inputCount, int outputCount);
}

public static class ActivationFunctions
{
    public abstract class ActivationFunctionBase : IActivationFunction
    {
        protected double[] Inputs = Array.Empty<double>();
        protected double[] Outputs = Array.Empty<double>();

        public double[] Squash(double[] inputs)
        {
            // save the inputs and outputs for training
            Inputs = new double[inputs.Length];
            inputs.CopyTo(Inputs, 0);
            Outputs = this.SquashImpl(inputs);
            return Outputs;
        }

        protected abstract double[] SquashImpl(double[] inputs);
        public abstract double[] BackProp(double[] errorWrtOutput);

        /**
        * Normalized Glorot Uniform Initializer
        * Glorot & Bengio, AISTATS 2010
        * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        */
        public virtual double[] WeightInitializers(int inputCount, int outputCount)
        {
            var rand = new Random();
            double limit = Math.Sqrt(6.0 / (inputCount + outputCount));
            return Enumerable.Range(0, inputCount)
                .Select(_ => 0.1 * rand.NextDouble(-limit, limit))
                .ToArray();
        }
    }

    public class LinearFunction : ActivationFunctionBase
    {
        protected override double[] SquashImpl(double[] inputs)
        {
            var outputs = new double[inputs.Length];
            inputs.CopyTo(outputs, 0);
            return outputs;
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            // the derivative of a constant is 1 so the output
            // becomes a copy of the supplied errors
            var pdOutputs = new double[Outputs.Length];
            errorWrtOutput.CopyTo(pdOutputs, 0);
            return pdOutputs;
        }
    }

    public class ReLuFunction : ActivationFunctionBase
    {
        protected override double[] SquashImpl(double[] inputs)
        {
            return inputs.Select(input => Math.Max(0.0, input)).ToArray();
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            return Inputs.Select((input, n) => errorWrtOutput[n] * ((input > 0) ? 1 : 0)).ToArray();
        }

        public override double[] WeightInitializers(int inputCount, int outputCount)
        {
            var rand = new Random();
            var std = Math.Sqrt(2.0 / inputCount);
            return Enumerable.Range(0, inputCount)
                .Select(_ => std * rand.NextDouble())
                .ToArray();
        }
    }

    public class LeakyReLuFunction : ActivationFunctionBase
    {
        private readonly double _a;

        // Initializes a new instance of the LeakyReLuFunction class.
        //
        // Parameters:
        //   a: The slope parameter for the Leaky ReLU function.       
        public LeakyReLuFunction(double a = 0.01)
        {
            _a = a;
        }

        protected override double[] SquashImpl(double[] inputs)
        {
            return inputs
                .Select(input => Math.Max(0.0, input) + _a * Math.Min(0.0, input))
                .ToArray();
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            return Inputs
                .Select((input, n) => errorWrtOutput[n] * ((input > 0) ? 1 : _a))
                .ToArray();
        }

        public override double[] WeightInitializers(int inputCount, int outputCount)
        {
            var rand = new Random();
            var std = Math.Sqrt(2.0 / inputCount);
            return Enumerable.Range(0, inputCount)
                .Select(_ => std * rand.NextDouble())
                .ToArray();
        }
    }

    /**
     * Sigmoid is a good activation function for binary classification
     */
    public class SigmoidFunction : ActivationFunctionBase
    {
        private readonly Func<double, double> _sigmoid = (x) => 1.0 / (1.0 + Math.Exp(-x));

        protected override double[] SquashImpl(double[] inputs)
        {
            return inputs.Select(input => _sigmoid(input)).ToArray();
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            return Outputs
                .Select(u => (1.0 - u) * u)
                .Zip(errorWrtOutput)
                .Select(p => p.First * p.Second)
                .ToArray();
        }
    }


    public class TanhFunction : ActivationFunctionBase
    {
        protected override double[] SquashImpl(double[] inputs)
        {
            return inputs.Select(input => Math.Tanh(input)).ToArray();
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            return Outputs.Select(output => 1.0 - Math.Pow(output, 2))
                .Select((d, n) => errorWrtOutput[n] * d)
                .ToArray();
        }
    }

    public class SoftmaxFunction : ActivationFunctionBase
    {
        protected override double[] SquashImpl(double[] inputs)
        {
            // subtract the maximum input from all the inputs
            // to prevent the exponential function from overflowing
            var maxInputValue = inputs.Max();
            var expValues = inputs
                .Select(i => i - maxInputValue)
                .Select(Math.Exp)
                .ToArray();

            var sum = expValues.Sum();
            return expValues.Select(v => v / sum).ToArray();
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            Debug.Assert(errorWrtOutput.Length == Outputs.Length,
                $"the length of the errors array ({errorWrtOutput.Length}) did not match the length of the output array ({Outputs.Length})");

            var derivatives = new double[Outputs.Length];

            for (var i = 0; i < Outputs.Length; i++) // row
            {
                var result = 0.0;
                for (var j = 0; j < Outputs.Length; j++) // column
                {
                    // partial derivative of the output at i,j
                    var pdij =
                        (i == j)
                            ? Outputs[i] * (1 - Outputs[i])
                            : -1.0 * Outputs[i] * Outputs[j];

                    result += pdij * errorWrtOutput[j];
                }

                derivatives[i] = result;
            }


            return derivatives;
        }
    }
}