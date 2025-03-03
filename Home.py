import streamlit as st
from PIL import Image
import base64

st.set_page_config(page_title="Home",
                   page_icon="🧊")

st.title("Sample Size Calculator",)

hide_st_style="""<style>
#MainMenu
{visiblility:hidden;
}
footer
{visibility: hidden;
}
header
{visibility: hidden;
}
</style>"""
st.markdown(hide_st_style,unsafe_allow_html=True)

#allow_output_mutation=True)
#def set_bg_hack(main_bg):
#    '''
#    A function to unpack an image from root folder and set as bg.
# 
#    Returns
#    -------
#    The background.
#    '''
#    # set bg name
#    main_bg_ext = "png"
#        
#    st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
#             background-size: cover
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
#
#set_bg_hack('')
image = Image.open('image.png')

st.image(image)

