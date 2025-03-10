import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

st.set_page_config(page_title="Proportion Estimation",
                   page_icon="🧊")

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


# Streamlit App
st.title("Sample Size Calculation for Proportion: Proportion Estimation")

## Functuion
def nSampleProp(p=0.5,d=0.05,Conf=0.95,designEf=1,dropOut=0):
    n= ((norm.ppf(1-(1-Conf)/2)/d)**2)*(p*(1-p))
    return(abs(round((n/(1-dropOut))*designEf)))

p = st.sidebar.number_input("Proportion (%)",value=50.0,min_value=0.0,max_value=100.0)
d = st.sidebar.number_input("Precision (%)",min_value=0.0, value=10.0,max_value=100.0)
ads= st.sidebar.radio("Choose Precision Option",options=['Absolute Precision','Relative to the Proportion'])

if(ads=='Absolute Precision'):
    d1=d
else:
    d1= ((d/100)*(p/100))*100

drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=100.0)

x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

if(x== "Given"):
    designEffect= st.sidebar.number_input("Design Effect", value=1.0,min_value=1.0,help= "values in integer. Minimum is 1")
    go= st.button("Calculate Sample Size")
else:
    m= st.sidebar.number_input("Number of cluster",min_value=2)
    ICC= st.sidebar.number_input("ICC",min_value=0.0)
    designEffect= 1+(m-1)*ICC
    col1,col2,col3=st.columns(3)
    col1.metric("Cluster Size (m)",value=m)
    col2.metric("Intra Class Correlation (ICC)",value=ICC)
    col3.metric("Design Effect",value= round(designEffect,2))
    go= st.button("Calculate Sample Size")

if go:
    confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
    out=[]

    for conf in confidenceIntervals:
        sample_size= nSampleProp(p=(p/100),d=(d1/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
        out.append(sample_size)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        "Sample Size": out
    })
    dds= nSampleProp(p=(p/100),d=(d1/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))
    if(ads=='Absolute Precision'):
        st.write(f"Asuming that **{(p)}%** of the individuals in the population exhibit the characteristic of interest, the study would need a sample size of:")
        st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <div style="
                font-size: 36px;
                font-weight: bold;
                background-color: #48D1CC;
                padding: 10px;
                border-radius: 10px;
                text-align: center;">
                {dds}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"participants to estimate the expected proportion with an absolute precision of **{(d1)}%** and **95%** confidence interval, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.")
    else:
        st.write(f"Asuming that **{(p)}%** of the individuals in the population exhibit the characteristic of interest, the study would need a sample size of:")
        st.markdown(f"""
            <div style="display: flex; justify-content: center;">
                <div style="
                font-size: 36px;
                font-weight: bold;
                background-color: #48D1CC;
                padding: 10px;
                border-radius: 10px;
                text-align: center;">
                {dds}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"participants to estimate the expected proportion with an absolute precision of **({(p)}% * {(d)}%) = {(d1)}%** and **95%** confidence interval, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.")

    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)


st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **Proportion-Based Sample Size Formula**")
st.latex(r"""
n = \left( \frac{Z_{1-\alpha/2}}{d} \right)^2 \times p (1 - p) \times DE
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( d \)**: Precision (margin of error).
- **\( p \)**: Expected proportion.
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of individuals per cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")

st.markdown("""
    <div style="
        background-color: #f9f871;
        padding: 10px;
        border-left: 5px solid orange;
        border-radius: 5px;
        font-size: 18px;">
        <b>Note:</b> The design effect option is only applicable when doing cluster random sampling, other wise the default is 1 and it is recommended to be done in consultation with a statistician.   
    </div>
    """, unsafe_allow_html=True)


st.subheader("📌 References")

st.markdown("""
1. **Naing, N. N. (2003).** Determination of Sample Size.The Malaysian Journal of Medical Sciences: MJMS,10(2), 84-86. Available at: [https://pubmed.ncbi.nlm.nih.gov/23386802/](https://pubmed.ncbi.nlm.nih.gov/23386802/)
""")

st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")