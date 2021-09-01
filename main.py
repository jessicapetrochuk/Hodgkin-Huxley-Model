import numpy as np
import streamlit as st
from scipy.integrate import odeint

from src.d01_modelling.neuron_constants import neuron_const
from src.d01_modelling.propogate_signal import integrate, input_current, steady_state_transition_state
from src.d01_modelling.dendrite import propogate_signal
from src.d02_visualization.streamlit_init import streamlit_init


def main():
    st.set_page_config(layout="wide")
    initial_voltage = -65
    voltage = st.sidebar.slider(label='Voltage of Stimulus', min_value=0, max_value=500)
    t_final = st.sidebar.slider(label="Total Time", min_value=0, max_value=200)
    length = st.sidebar.slider(label="Length of Neuron", min_value=0, max_value=200)
    T = np.linspace(0, t_final, 1000)
    stimulus = [input_current(x, voltage) for x in T]
    
    Y = np.array([
        initial_voltage, 
        steady_state_transition_state(initial_voltage, 'n'), 
        steady_state_transition_state(initial_voltage, 'm'),
        steady_state_transition_state(initial_voltage, 'h')
    ])

    
    integrals = odeint(integrate, Y, T, args=(voltage,))
    voltage = integrals[:, 0]
    n = integrals[:, 1]
    m = integrals[:, 2]
    h = integrals[:, 3]
    propogated_signal = propogate_signal(voltage, length)
    streamlit_init(np.fromiter(stimulus, dtype=int), voltage, n, m, h, T, propogated_signal, length)


if __name__ == '__main__':
    main()