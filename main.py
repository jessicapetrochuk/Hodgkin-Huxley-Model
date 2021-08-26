import numpy as np
import streamlit as st
from scipy.integrate import odeint

from src.d01_modelling.neuron_constants import neuron_const
from src.d01_modelling.propogate_signal import integrate, input_current, steady_state_transition_state
from src.d01_modelling.dendrite import propogate_signal
from src.d02_visualization.streamlit_init import streamlit_init


def main():
    initial_voltage = -65
    t_final = 150 #ms
    st.set_page_config(layout="wide")
    voltage = st.slider(label='Voltage of stimulus', min_value=0, max_value=500)
    stimulus = [input_current(x, voltage) for x in range(t_final)]
    Y = np.array([
        initial_voltage, 
        steady_state_transition_state(initial_voltage, 'n'), 
        steady_state_transition_state(initial_voltage, 'm'),
        steady_state_transition_state(initial_voltage, 'h')
    ])
    T = np.array([x for x in range(t_final)])
    integrals = odeint(integrate, Y, T, args=(voltage,))
    voltage = integrals[:, 0]
    n = integrals[:, 1]
    m = integrals[:, 2]
    h = integrals[:, 3]

    propogated_1, propogated_2, propogated_3 = propogate_signal(voltage, neuron_const['C_m'], neuron_const['g_m'],neuron_const['lambda_m'])
    streamlit_init(np.fromiter(stimulus, dtype=int), voltage, n, m, h, propogated_1, propogated_2, propogated_3)


if __name__ == '__main__':
    main()