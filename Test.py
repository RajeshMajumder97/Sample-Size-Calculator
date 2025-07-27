import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="centered")

# Display Intro.js launcher
components.html(
    """
    <link href="https://cdn.jsdelivr.net/npm/intro.js/minified/introjs.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/intro.js/minified/intro.min.js"></script>

    <style>
        .introjs-helperLayer {
            background-color: rgba(255, 255, 255, 0.3) !important;
            backdrop-filter: blur(3px);
        }
        .custom-highlight {
            border: 3px solid #007BFF;
            border-radius: 8px;
            padding: 4px;
        }
        #guide-launch {
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #f0f0f0;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
    </style>

    <button id="guide-launch" onclick="startIntro()">🚀 Start Guide</button>

    <script>
    function startIntro(){
        introJs().setOptions({
            highlightClass: 'custom-highlight',
            overlayOpacity: 0.5,
            steps: [
                {
                    intro: "👋 Welcome to the Sample Size Calculator!"
                },
                {
                    element: document.querySelector('#study_id'),
                    intro: "Choose your study design here.",
                    position: "right"
                },
                {
                    element: document.querySelector('#alpha_id'),
                    intro: "Set your significance level (commonly 0.05).",
                    position: "right"
                },
                {
                    element: document.querySelector('#power_id'),
                    intro: "Choose statistical power, usually 0.8 or 0.9.",
                    position: "right"
                },
                {
                    element: document.querySelector('#calc_id'),
                    intro: "Click to calculate sample size!",
                    position: "right"
                }
            ]
        }).start();
    }
    </script>
    """,
    height=150,
)

# Now the actual form
st.markdown('<div id="study_id"><label><strong>Study Design</strong></label></div>', unsafe_allow_html=True)
study = st.selectbox("", ["Case-Control", "Cohort", "ICC"], key="study_design")

st.markdown('<div id="alpha_id"><label><strong>Significance Level (α)</strong></label></div>', unsafe_allow_html=True)
alpha = st.number_input("", min_value=0.01, max_value=0.1, value=0.05, step=0.01, key="alpha_val")

st.markdown('<div id="power_id"><label><strong>Power (1−β)</strong></label></div>', unsafe_allow_html=True)
power = st.number_input("", min_value=0.7, max_value=0.99, value=0.8, step=0.01, key="power_val")

st.markdown('<div id="calc_id"></div>', unsafe_allow_html=True)
if st.button("Calculate"):
    st.success(f"✅ Sample size calculated for {study} with α={alpha}, power={power}")
