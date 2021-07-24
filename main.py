import numpy as np
from scipy.integrate import odeint
from src.d02_visualization.streamlit_init import streamlit_init
from src.d01_modelling.neuron import Model

def main(Model):
    stimulus = map(Model.input_current, range(Model.t_final))
    Y = np.array([-70, Model.n_inf(), Model.m_inf(), Model.h_inf()])
    T = np.linspace(0, Model.t_final)
    integrals = odeint(Model.integrate_, Y, T)
    streamlit_init(np.fromiter(stimulus, dtype=int), integrals[:, 0])


if __name__ == '__main__':
    model = Model()
    main(model)