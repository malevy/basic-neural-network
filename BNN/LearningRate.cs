namespace BNN;

public class LearningRate
{
    private readonly double _learningRate;
    private readonly double _decay;
    private int _iteration;

    public LearningRate(double learningRate, double decay = 0.0)
    {
        _learningRate = learningRate;
        _decay = decay;
    }

    public void Decay()
    {
        _iteration++;
    }
    
    public double Value => _learningRate * (1.0 / (1.0 + (_decay * _iteration)));
}