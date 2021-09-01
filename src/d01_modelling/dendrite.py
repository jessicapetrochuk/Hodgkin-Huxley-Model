## For a dendrite infinitely long in either direction
import math
from src.d01_modelling.neuron_constants import neuron_const


def propogate_signal(V_df, distance):
    """Calculates what the signal would be after 3 different lengths using the cable equation
    Args:
        V_df (np.array): membrane voltage passing through the neuron
    """
    propogated_signal = [
        v * math.exp(-abs(distance) / neuron_const["lambda_m"]) for v in V_df
    ]
    return propogated_signal
