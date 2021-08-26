import math

from src.d01_modelling.propogate_signal import steady_state_transition_state, transition_rate, transition_state_change

def test_transition_rate_alpha_n():
    V = 100
    str = 'alpha_n'
    assert transition_rate(V, str) == (0.01 * (10 - 100)) / (math.exp((10 - 100) / 10) - 1)

def test_transition_rate_beta_n():
    V = 100
    str = 'beta_n'
    assert transition_rate(V, str) == 0.125 * math.exp(-100 / 80)

def test_transition_rate_alpha_m():
    V = 100
    str = 'alpha_m'
    assert transition_rate(V, str) == (0.1 * (25 - 100)) / (math.exp((25 - 100) / 10) - 1)

def test_transition_rate_beta_m():
    V = 100
    str = 'beta_m'
    assert transition_rate(V, str) == 4 * math.exp(-100 / 18)

def test_transition_rate_alpha_h():
    V = 100
    str = 'alpha_h'
    assert transition_rate(V, str) == 0.07 * math.exp(-100 / 20)

def test_transition_rate_beta_h():
    V = 100
    str = 'beta_h'
    assert transition_rate(V, str) == 1 / (math.exp((30 - 100) / 10) + 1)

def test_transition_state_change():
    trans_rate_alpha = 2
    trans_rate_beta = 5
    x = 0.1
    assert transition_state_change(trans_rate_alpha, trans_rate_beta, x) == (2* (1.0 - 0.1)) - (5 * 0.1)

def test_steady_state_transition_state():
    V = 100
    x = 'n'
    alpha_n = (0.01 * (10 - 100)) / (math.exp((10 - 100) / 10) - 1)
    beta_n = 0.125 * math.exp(-100 / 80)
    assert steady_state_transition_state(V, x) == alpha_n / (alpha_n + beta_n)