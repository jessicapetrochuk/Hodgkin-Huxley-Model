import math

import numpy as np

from src.d01_modelling.neuron_constants import neuron_const


transition_rate_equations = {
    "alpha_n": lambda V: (0.02 * (V - 25)) / (1 - math.exp((25 - V) / 9)),
    "beta_n": lambda V: (-0.002 * (V - 25)) / (1 - math.exp((V - 25) / 9)),
    "alpha_m": lambda V: (0.182 * (V + 35)) / (1 - math.exp((-35 - V) / 9)),
    "beta_m": lambda V: (-0.124 * (V + 35)) / (1 - math.exp((35 + V) / 9)),
    "alpha_h": lambda V: 0.25 * math.exp((-90 - V) / 12),
    "beta_h": lambda V: (0.25 * math.exp((V + 62) / 6)) / (math.exp((V + 90) / 12)),
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
    return neuron_const["g_na_max"] * h * m ** 3


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


def channel_current(V, g, E) -> float:
    """Calculates present current through channel
    Args:
        V (float): voltage
        g (float): conductance through channgel
        E (float): nernst potential of channel
    """
    return g * (V - E)


def input_current(t, max_current) -> float:
    """Returns the input current at a given time t
    Args:
        t (int): current time in sec
        max_current (int): current during neuronal propogation
    """
    if t >= 20 and t <= 21:
        return max_current
    return 0


def integrate(y, t, current) -> np.array:
    """Creates a set of ordinary differential equations
    Args:
        y (np.array): present state of voltage, n, m, and h
        t (float): present time
        current (int): present amount of current flowing through the neuron
    """
    integrals = np.zeros((4,))

    V = y[0]
    n = y[1]
    m = y[2]
    h = y[3]

    i_na = channel_current(V, g_na(m, h), neuron_const["E_na"])
    i_k = channel_current(V, g_k(n), neuron_const["E_k"])
    i_l = channel_current(V, g_l(), neuron_const["E_l"])
    i_tot = i_na + i_k + i_l

    integrals[0] = (input_current(t, current) - i_tot) / neuron_const["C_m"]

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
