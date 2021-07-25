import numpy as np
from scipy.integrate import odeint
from src.d02_visualization.streamlit_init import streamlit_init
from src.d01_modelling.neuron import Model
import streamlit as st

def main(Model):
    st.set_page_config(layout="wide")
    voltage = st.slider(label='Voltage of stimulus', min_value=0, max_value=500)
    stimulus = [Model.input_current(x, voltage) for x in range(Model.t_final)]
    Y = np.array([-70, Model.n_inf(voltage), Model.m_inf(voltage), Model.h_inf(voltage)])
    T = np.linspace(0, Model.t_final)
    integrals = odeint(Model.integrate_, Y, T, args=(voltage,))
    voltage = integrals[:, 0]
    n = integrals[:, 1]
    m = integrals[:, 2]
    h = integrals[:, 3]
    streamlit_init(np.fromiter(stimulus, dtype=int), voltage, n, m, h)


if __name__ == '__main__':
    model = Model()
    main(model)