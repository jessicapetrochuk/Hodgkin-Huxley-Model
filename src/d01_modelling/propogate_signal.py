import math

import numpy as np

from src.d01_modelling.neuron_constants import neuron_const

transition_rate_equations = {
    "alpha_n": lambda V: (0.01 * (10 - V)) / (math.exp((10 - V) / 10) - 1),
    "beta_n": lambda V: 0.125 * math.exp(-V / 80),
    "alpha_m": lambda V: (0.1 * (25 - V)) / (math.exp((25 - V) / 10) - 1),
    "beta_m": lambda V: 4 * math.exp(-V / 18),
    "alpha_h": lambda V: 0.07 * math.exp(-V / 20),
    "beta_h": lambda V: 1 / (math.exp((30 - V) / 10) + 1),
}


def transition_rate(V, trans_rate) -> float:
    """Calculates transition rate
    Args:
        V (float): current membrane voltage
        trans_rate (str): desired transition rate
    """
    return transition_rate_equations[trans_rate](V)


def transition_state_change(trans_rate_alpha, trans_rate_beta, x) -> float:
    """Calculates change in probability of channel being open
    Args:
        V (float): current membrane voltage
        trans_rate (float): transition rate
        x (float): probability of channel being open
    """
    return (trans_rate_alpha * (1.0 - x)) - (trans_rate_beta * x)


def g_na(m, h) -> float:
    """Calculates present conductance through sodium channel
    Args:
        m (float): probability of sodium channel being open
        h (float): probability of sodium channel inactivation gate being open
    """
    return neuron_const["g_na_max"] * m ** 3 * h


def g_k(n) -> float:
    """Calculates present conductance through potassium channel
    Args:
        n (float): probability of potassium channel being open
    """
    return neuron_const["g_k_max"] * n ** 4


def g_l() -> float:
    """Returns present conductance through the leak channel"""
    return neuron_const["g_l_max"]


def steady_state_transition_state(V, trans_rate) -> float:
    """Calculates steady state transition state
    Args:
        V (float): voltage
        trans_rate (str): desired transition rate
    """
    alpha_x = transition_rate_equations[f"alpha_{trans_rate}"](V)
    beta_x = transition_rate_equations[f"beta_{trans_rate}"](V)
    return alpha_x / (alpha_x + beta_x)


def current(V, g, E):
    """Calculates present current through channel
    Args:
        V (float): voltage
        g (float): conductance through channgel
        E (float): nernst potential of channel
    """
    return g * (V - E)


def input_current(t, max_current):
    """Returns the input current at a given time t
    Args:
        t (int): current time in sec
        max_current (int): max_current during neuronal propogation
    """
    if 15 <= t <= 16 or 40 <= t <= 41:
        return max_current
    return 0


def integrate(y, t_0, max_current):
    integrals = np.zeros((4,))

    V = y[0]
    n = y[1]
    m = y[2]
    h = y[3]

    i_na = current(V, g_na(m, h), neuron_const["E_na"])
    i_k = current(V, g_k(n), neuron_const["E_k"])
    i_l = current(V, g_l(), neuron_const["E_l"])
    i_tot = i_na + i_k + i_l

    integrals[0] = (input_current(t_0, max_current) - i_tot) / neuron_const["C_m"]

    alpha_n = transition_rate(V, "alpha_n")
    beta_n = transition_rate(V, "beta_n")

    alpha_m = transition_rate(V, "alpha_m")
    beta_m = transition_rate(V, "beta_m")

    alpha_h = transition_rate(V, "alpha_h")
    beta_h = transition_rate(V, "beta_h")

    integrals[1] = transition_state_change(alpha_n, beta_n, n)
    integrals[2] = transition_state_change(alpha_m, beta_m, m)
    integrals[3] = transition_state_change(alpha_h, beta_h, h)

    return integrals
