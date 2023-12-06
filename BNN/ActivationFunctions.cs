namespace BNN;

public interface IActivationFunction
{
    double[] Squash(double[] inputs);
    double[] BackProp(double[] errorWrtOutput);
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
    }

    public class SigmoidFunction : ActivationFunctionBase
    {
        private readonly Func<double, double> _sigmoid = (x) => 1.0 / (1.0 + Math.Exp(-x));

        protected override double[] SquashImpl(double[] inputs)
        {
            return inputs.Select(input => _sigmoid(input)).ToArray();
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            return Inputs
                .Select(input => _sigmoid(input) * (1 - _sigmoid(input)))
                .Select((d, n) => errorWrtOutput[n] * d)
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
            return Outputs.Select(output => 1.0 - Math.Pow(Math.Tanh(output), 2))
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
            var derivatives = new double[Outputs.Length];

            for (var i = 0; i < Outputs.Length; i++) // row
            {
                var result = 0.0;
                for (var j = 0; j < Outputs.Length; j++) // column
                {
                    result +=
                        (i == j)
                            ? Outputs[i] * (1 - Outputs[i])
                            : -1.0 * Outputs[i] * Outputs[j];
                }

                derivatives[i] = result;
            }

            var errorWrtNet = derivatives
                .Zip(errorWrtOutput)
                .Select(x => x.First * x.Second)
                .ToArray();
            
            return errorWrtNet;
        }
    }

}