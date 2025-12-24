import streamlit as st
import numpy as np
import pandas as pd

def tampilan_proyek():
    data = pd.DataFrame(np.random.randn(50, 3), columns=['A', 'B', 'C'])
    st.line_chart(data)

    st.markdown("### Filter Data")
    range_slider = st.slider("Pilih range nilai:", 0, 100, (25, 75))
    st.write(f"Anda memilih range: {range_slider}")