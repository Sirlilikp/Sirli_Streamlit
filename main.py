import streamlit as st
import prediksi

st.set_page_config(layout='wide')
st.title('Portfolio Saya')
st.header('Data Scientist')

st.sidebar.title('navigasi')

page = st.sidebar.radio('Pilih halaman', ['Tentang Proyek', 
                                          'Proyek', 'Prediction', 
                                          'Kontak'])

if page == 'Tentang Proyek':
    import tentang
    tentang.about_me()
elif page == 'Proyek':
    import proyek
    proyek.tampilan_proyek()
elif page == 'Kontak':
    import kontak
    kontak.munculkan_kontak()
elif page == "Prediction":
    prediksi.app() 