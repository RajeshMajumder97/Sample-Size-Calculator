import streamlit as st
from PIL import Image
import base64
from io import BytesIO

def inject_logo(width=140):
    logo = Image.open("image2.png")

    def image_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    logo_base64 = image_to_base64(logo)

    st.markdown(
        f"""
        <style>
        /* Reduce default sidebar padding */
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }}

        /* Container to hold the logo cleanly */
        .custom-logo {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 0.75rem;
            width: {width}px;
        }}
        </style>

        <div class="custom-logo">
            <img src="data:image/png;base64,{logo_base64}" width="{width}" />
        </div>
        """,
        unsafe_allow_html=True
    )
