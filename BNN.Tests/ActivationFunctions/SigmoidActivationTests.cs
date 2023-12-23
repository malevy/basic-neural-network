namespace BNN.Tests.ActivationFunctions;

public class SigmoidActivationTests
{

    [Test]
    public void PositiveValueTests()
    {
        var activation = new BNN.ActivationFunctions.SigmoidFunction();
        Assert.That(activation.Squash(new[] { 2.0 })[0], Is.EqualTo(0.88).Within(0.01));
        Assert.That(activation.Squash(new[] { 3.0 })[0], Is.EqualTo(0.95).Within(0.01));
    }
    
    [Test]
    public void NegativeValueTests()
    {
        var activation = new BNN.ActivationFunctions.SigmoidFunction();
        Assert.That(activation.Squash(new[] { -2.0 })[0], Is.EqualTo(0.12).Within(0.01));
        Assert.That(activation.Squash(new[] { -3.0 })[0], Is.EqualTo(0.05).Within(0.01));
    }

    [Test]
    public void DerivativeTests()
    {
        var actFunc = new WrappedSigmoidFunction();
        actFunc.SetOutputs(new double[] {1.0, 2.0});
        var errorToProp = actFunc.BackProp(new double[] { 3.0, 4.0 });
        Assert.That(errorToProp[0], Is.EqualTo(0.0).Within(0.01));
        Assert.That(errorToProp[1], Is.EqualTo(-8.0).Within(0.01));
    }

    private class WrappedSigmoidFunction : BNN.ActivationFunctions.SigmoidFunction
    {
        public void SetOutputs(double[] outputs)
        {
            this.Outputs = outputs;
        }
    }
    
    
}