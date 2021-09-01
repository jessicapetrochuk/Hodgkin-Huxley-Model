from altair.vegalite.v4.schema.channels import StrokeDash
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import altair as alt


def streamlit_init(
    stimulus_df, membrane_potential_df, n_df, m_df, h_df, T, propogated_signal, length
):
    stimulus_df = pd.DataFrame({"Current Input": stimulus_df, "Time": T})
    membrane_voltage_df = pd.DataFrame(
        {"Membrane Voltage": membrane_potential_df, "Time": T}
    )
    activation_variables_df = pd.DataFrame({"Time": T, "n": n_df, "m": m_df, "h": h_df})
    propogated_signal_df = pd.DataFrame(
        {"Voltage after {} mm".format(str(length)): propogated_signal, "Time": T}
    )
    stimulus_chart = (
        alt.Chart(stimulus_df)
        .mark_line()
        .encode(x="Time", y="Current Input")
        .properties(title="Stimulus vs Time")
    )
    membrane_voltage_chart = (
        alt.Chart(membrane_voltage_df)
        .mark_line()
        .encode(x="Time", y="Membrane Voltage")
        .properties(title="Membrane Potential vs. Time")
    )
    activation_variables_chart = (
        alt.Chart(activation_variables_df)
        .transform_fold(["n", "m", "h"], as_=["Key", "Voltage"])
        .mark_line()
        .encode(x="Time", y="Voltage:Q", color="Key:N")
        .properties(title="Activation/Inactivation Variables vs. Time")
    )
    propogated_signal_chart = (
        alt.Chart(propogated_signal_df)
        .mark_line()
        .encode(x="Time", y="Voltage after {} mm".format(str(length)))
        .properties(title="Voltage Propogation Down Dendrite Over {} mm".format(length))
    )

    st.title("Hodkin Huxley Model")

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(stimulus_chart)
    with col2:
        st.altair_chart(membrane_voltage_chart)

    col3, col4 = st.columns(2)
    with col3:
        st.altair_chart(activation_variables_chart)
    with col4:
        st.altair_chart(propogated_signal_chart)
