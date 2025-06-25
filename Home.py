import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Load logo image (your uploaded icon)
logo = Image.open("image.png")

# Helper function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

logo_base64 = image_to_base64(logo)


# Centered title and subtitle
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 68px;'>StudySizer</h1>
        <h3 style='font-size: 38px;'>A Sample Size Calculator</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("This open-source and free web application allows researchers, students, and professionals to calculate"
"the required sample size for their studies. It offers a user-friendly interface and supports a range of statistical"
" methods for different study designs. The advantage of this tool is, it also gives the required sample sie calculation formulas along with the references.")

st.markdown("Hi, I am Rajesh, a Ph.D. student in Biostatistics. If you find this tool useful, please cite it as:")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.streamlit.app/](https://studysizer.streamlit.app/))*")

st.markdown("**If you want to reach me :**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")


# Inject CSS and image at the top of the sidebar
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 80px;
        background-image: url("data:image/png;base64,{logo_base64}");
        background-repeat: no-repeat;
        background-position: 20px 20px;
        background-size: 50px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.image("Sample size Explained.png")
st.image("Sample size Explained_2.png")