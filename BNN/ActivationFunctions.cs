namespace BNN;

public interface IActivationFunction
{
    double Squash(double net);
    double PartialDee(double net, double outValue);
} 

public static class ActivationFunctions
{

    private class LinearFunction : IActivationFunction
    {
        public double Squash(double net) => net;
        public double PartialDee(double net, double outValue) => 1;
    }
    public static IActivationFunction Linear = new LinearFunction();

    private class ReLUFunction : IActivationFunction
    {
        public double Squash(double net) => Math.Max(0, net);
        public double PartialDee(double net, double outValue) => (net > 0) ? 1 : 0; 
    }
    public static IActivationFunction ReLU = new ReLUFunction();

    private class SigmoidFunction : IActivationFunction
    {
        public double Squash(double net) => 1.0 / (1.0 + Math.Exp(-net));
        public double PartialDee(double net, double outValue) => Squash(net) * (1.0 - Squash(net)); 
    }

    public static IActivationFunction Sigmoid = new SigmoidFunction();

    private class TanhFunction : IActivationFunction
    {
        public double Squash(double net) => Math.Tanh(net);
        public double PartialDee(double net, double outValue) => 1.0 - Math.Pow( Math.Tanh(outValue), 2);
        // public double PartialDee(double net, double outValue) => 1.0 - Math.Pow( Math.Tanh(outValue), 2);
    }

    public static IActivationFunction Tanh = new TanhFunction();
}