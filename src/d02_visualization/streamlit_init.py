import streamlit as st
import pandas as pd

def streamlit_init(stimulus_df, membrane_potential_df, n_df, m_df, h_df):
    membrane_voltage_df = pd.DataFrame({"Membrane Voltage": membrane_potential_df})
    stimulus_df = pd.DataFrame({"Stimulus": stimulus_df})
    n_df = pd.DataFrame({"n": n_df, 'm': m_df, 'h': h_df})
    st.title('Hodkin Huxley Model')
    st.write('Stimulus vs Time')
    st.line_chart(stimulus_df)
    st.write('Membrane Potential vs. Time')
    st.line_chart(membrane_voltage_df)
    st.write('Activation/Inactivation Variables vs. Time')
    st.line_chart(n_df)