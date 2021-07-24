import streamlit as st
import pandas as pd

def streamlit_init(stimulus_df, membrane_potential_df):
    st.set_page_config(layout="wide")
    membrane_voltage_df = pd.DataFrame({"Membrane Voltage": membrane_potential_df})
    stimulus_df = pd.DataFrame({"Stimulus": stimulus_df})
    st.title('Hodkin Huxley Model')
    st.write('Stimulus vs Time Graph')
    stimulus_chart = st.line_chart(stimulus_df)
    st.write('Membrane Potential vs. Time Graph')
    membrane_voltage_chart = st.line_chart(membrane_voltage_df)