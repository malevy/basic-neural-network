namespace BNN.Tests.ActivationFunctions;

public class ActivationFunctionBaseTests
{

    [Test]
    public void InputsAreCapturedProperly()
    {
        var func = new PassThroughFunction();
        var inputs = new[] { 1.0, 2.0, 3.0 };
        func.Squash(inputs);
        
        Assert.That(func.GetInputs, Is.EqualTo(inputs));
        Assert.That(func.GetOutputs, Is.EqualTo(inputs.Select(i => i*2).ToArray()));
    }

    class PassThroughFunction : BNN.ActivationFunctions.ActivationFunctionBase
    {
        protected override double[] SquashImpl(double[] inputs)
        {
            return inputs.Select(i => i*2).ToArray();
        }

        public override double[] BackProp(double[] errorWrtOutput)
        {
            throw new NotImplementedException();
        }

        public double[] GetOutputs => this.Outputs;
        public double[] GetInputs => this.Inputs;
    }
}
