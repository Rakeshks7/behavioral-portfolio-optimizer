import numpy as np
import pandas as pd
import pytest
from src.behavioral_optimizer import BehavioralOptimizer

@pytest.fixture
def sample_returns():
    np.random.seed(42)
    data = np.random.normal(loc=0.0005, scale=0.015, size=(100, 4))
    df = pd.DataFrame(data, columns=['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D'])
    return df

@pytest.fixture
def optimizer(sample_returns):
    return BehavioralOptimizer(returns=sample_returns, risk_free_rate=0.0)

def test_initialization(optimizer):
    assert optimizer.alpha == 0.88
    assert optimizer.beta == 0.88
    assert optimizer.lambda_ == 2.25
    assert optimizer.n_assets == 4

def test_value_function_loss_aversion(optimizer):
    gain = np.array([0.05])
    loss = np.array([-0.05])
    
    v_gain = optimizer._value_function(gain)[0]
    v_loss = optimizer._value_function(loss)[0]

    assert abs(v_loss) > v_gain
    assert v_gain > 0
    assert v_loss < 0

def test_probability_weighting_bounds(optimizer):
    p_values = np.array([0.0, 0.1, 0.5, 0.9, 1.0])

    w_gains = optimizer._probability_weight(p_values, optimizer.gamma_plus)
    assert np.all(w_gains >= 0.0)
    assert np.all(w_gains <= 1.0)

    w_losses = optimizer._probability_weight(p_values, optimizer.gamma_minus)
    assert np.all(w_losses >= 0.0)
    assert np.all(w_losses <= 1.0)

def test_optimizer_constraints(optimizer):
    for method in ['prospect', 'sharpe']:
        result = optimizer.optimize(method=method)
        weights = result['weights']

        assert result['success'] is True

        assert len(weights) == optimizer.n_assets

        assert pytest.approx(np.sum(weights), abs=1e-4) == 1.0

        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)