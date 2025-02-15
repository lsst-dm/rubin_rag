import streamlit as st


def setup_sidebar():
    st.sidebar.markdown("Select sources to search:")
    st.session_state["required_sources"] = []
    if st.sidebar.checkbox("Confluence", value=True):
        st.session_state["required_sources"].append("confluence")
    if st.sidebar.checkbox("Jira", value=True):
        st.session_state["required_sources"].append("jira")
    if st.sidebar.checkbox("LSST Forum Docs", value=True):
        st.session_state["required_sources"].append("lsstforum")
    if st.sidebar.checkbox("Local Docs", value=True):
        st.session_state["required_sources"].append("localdocs")


def setup_landing_page():
    # Display the landing page until the first message is sent
    if not st.session_state.message_sent:
        with st.container():
            # Add logo (Make sure the logo is in your working directory or provide the full path)
            st.image("./static/rubin_avatar_bw.png", clamp=True)

            # Centered title and message
            st.markdown(
                "<h2 class='h2-landing-page'>Hello, I'm Vera!</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h4 class='h4-landing-page'>Your dedicated Rubin Observatory bot.</h4>",
                unsafe_allow_html=True,
            )


def setup_header_and_footer(msgs):
    def clear_text():
        msgs.clear()
        st.session_state.message_sent = False

    st.button(":material/edit_square:", on_click=clear_text)
    st.markdown(
        "<footer class='footer-fixed'>Vera aims for clarity, but can make mistakes.</footer>",
        unsafe_allow_html=True,
    )
