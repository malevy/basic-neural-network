namespace BNN.Tests.ActivationFunctions;

public class SoftmaxActivationTests
{

    
    [Test]
    public void SquashHappyPath()
    {
        var inputs = new[] { 4.8, 1.21, 2.385 };
        var expected = new[] { 0.8952826639573506, 0.024708306782070668, 0.08000902926057876 };

        var actFunc = new BNN.ActivationFunctions.SoftmaxFunction();
        var actual = actFunc.Squash(inputs);
        
        Assert.That(actual, Is.EqualTo(expected).Within(0.001));
    }

    [Test]
    public void DerivativeHappyPath()
    {
        var inputs = new[] { 0.7, 0.1, 0.2 };
        var expected = new[] {0, 0,0 };
        var wrapper = new SoftmaxActivationWrapper();
        wrapper.SetOutputs(inputs);

        var actual = wrapper.BackProp(new[] { 1.0, 1.0, 1.0 });
        Console.WriteLine(String.Join(',',actual));
        Assert.That(actual, Is.EqualTo(expected).Within(0.00001));
    }

    /*
     * this wrapper gives our tests access to the protected Output property without
     * figuring out a set of inputs that gets to the desired outputs when testing
     * the derivative
     */
    private class SoftmaxActivationWrapper : BNN.ActivationFunctions.SoftmaxFunction
    {
        public void SetOutputs(double[] outputs)
        {
            this.Outputs = outputs;
        }
    }
    
    
}