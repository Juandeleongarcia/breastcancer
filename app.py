"""BreastDCEDL AI: ejecutar con python -m streamlit run app.py."""
import streamlit as st

st.set_page_config(page_title='BreastDCEDL AI · Treatment Response',page_icon=':material/biotech:',
                   layout='wide',initial_sidebar_state='auto')

from app_components import home,explorer,prediction,validation,methodology
from app_components.layout import initialize,load_style,sidebar,topbar,notice


def main():
    initialize()
    load_style()
    sidebar()
    page = st.session_state.page
    topbar()
    if st.session_state.get('settings_error'):st.warning(st.session_state.settings_error)
    if page=='home':home.render()
    elif page=='explorer':explorer.render()
    elif page=='predict':prediction.render()
    elif page=='results':validation.render(st.session_state.workspace.results_campaign)
    elif page=='method':methodology.render()
    notice()


if __name__=='__main__':main()
