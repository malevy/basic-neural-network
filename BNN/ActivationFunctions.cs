namespace BNN;

public interface IActivationFunction
{
    double Squash(double net);
    double PartialDee(double net, double outValue);

    double[] Squash(double[] inputs);
    double[] BackProp(double[] errorWrtOutput);
} 

public static class ActivationFunctions
{
    public abstract class ActivationFunctionBase : IActivationFunction
    {
        protected double[] Inputs = Array.Empty<double>();
        protected double[] Outputs = Array.Empty<double>();
        
        public abstract double Squash(double net);
        public abstract double PartialDee(double net, double outValue);

        public double[] Squash(double[] inputs)
        {
            Inputs = new double[inputs.Length];
            inputs.CopyTo(Inputs,0);
            Outputs = this.SquashImpl(inputs);
            return Outputs;
        }

        protected abstract double[] SquashImpl(double[] inputs);
        public abstract double[] BackProp(double[] errorWrtOutput);
    }

    public class LinearFunction : ActivationFunctionBase
    {
        public override double Squash(double net) => net;
        public override double PartialDee(double net, double outValue) => 1;
        
        protected override double[] SquashImpl(double[] inputs)
        {
            var outputs = new double[inputs.Length];
            inputs.CopyTo(outputs,0);
            return outputs;
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            // the derivative of a constant is 1 so the output
            // becomes a copy of the supplied errors
            var pdOutputs = new double[Outputs.Length];
            errorWrtOutput.CopyTo(pdOutputs,0);
            return pdOutputs;
        }
    }

    public class ReLuFunction : ActivationFunctionBase
    {
        public override double Squash(double net) => Math.Max(0, net);
        public override double PartialDee(double net, double outValue) => (net > 0) ? 1 : 0;
        
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
        
        public override double Squash(double net) => 1.0 / (1.0 + Math.Exp(-net));
        public override double PartialDee(double net, double outValue) => Squash(net) * (1.0 - Squash(net));
        
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
        public override double Squash(double net) => Math.Tanh(net);
        public override double PartialDee(double net, double outValue) => 1.0 - Math.Pow( Math.Tanh(outValue), 2);

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
}