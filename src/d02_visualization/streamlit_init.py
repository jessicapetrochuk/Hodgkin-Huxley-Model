import streamlit as st
import pandas as pd


def streamlit_init(
    stimulus_df, membrane_potential_df, n_df, m_df, h_df, v_df_1, v_df_2, v_df_3
):
    membrane_voltage_df = pd.DataFrame({"Membrane Voltage": membrane_potential_df})
    stimulus_df = pd.DataFrame({"Stimulus": stimulus_df})
    activation_variables_df = pd.DataFrame({"n": n_df, "m": m_df, "h": h_df})
    voltage_propogation_df = pd.DataFrame(
        {
            "Voltage after 1mm": v_df_1,
            "Voltage after 5mm": v_df_2,
            "Voltage after 10mm": v_df_3,
        }
    )
    st.title("Hodkin Huxley Model")
    st.write("Stimulus vs Time")
    st.line_chart(stimulus_df)
    st.write("Membrane Potential vs. Time")
    st.line_chart(membrane_voltage_df)
    st.write("Activation/Inactivation Variables vs. Time")
    st.line_chart(activation_variables_df)
    st.write("Voltage Propogation Down Dendrite Over Length x")
    st.line_chart(voltage_propogation_df)
