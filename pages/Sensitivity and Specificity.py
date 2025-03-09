import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
import math

st.set_page_config(page_title="Sensitivity and Specificity",
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
st.title("Sample Size Calculation for Sensitivity & Specificity: Estimation")

## Functuion
def nSampleSen(p=0.5,Se=0.80,d=0.05,Conf=0.95,designEf=1,dropOut=0):
    n= ((norm.ppf(1-(1-Conf)/2)/(d*math.sqrt(p)))**2)*(Se*(1-Se))
    return(abs(round((n/(1-dropOut))*designEf)))

def nSampleSpc(p=0.5,Sp=0.70,d=0.05,Conf=0.95,designEf=1,dropOut=0):
    n= ((norm.ppf(1-(1-Conf)/2)/(d*math.sqrt(1-p)))**2)*(Sp*(1-Sp))
    return(round((n/(1-dropOut))*designEf))

p = st.sidebar.number_input("Prevalence of the Event (%)",value=50.0,min_value=0.0,max_value=100.0)
Se = st.sidebar.number_input("Anticipated Sensitivity (%)",value=80.0,min_value=0.0,max_value=100.0)
Sp = st.sidebar.number_input("Anticipated Specificity (%)",value=70.0,min_value=0.0,max_value=100.0)
d = st.sidebar.number_input("Precision (%)", value=5.0,min_value=0.0,max_value=100.0)
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
    out1=[]
    out2=[]

    for conf in confidenceIntervals:
        sample_size_sen= nSampleSen(p=(p/100),Se=(Se/100),d=(d/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
        sample_size_spc= nSampleSpc(p=(p/100),Sp=(Sp/100),d=(d/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
        out1.append(sample_size_sen)
        out2.append(sample_size_spc)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        " Sensitivity Sample Size": out1,
        " Specificity Sample Size": out2
    })

    dds1= nSampleSen(p=(p/100),Se=(Se/100),d=(d/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))
    dds2= nSampleSpc(p=(p/100),Sp=(Sp/100),d=(d/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))
    
    st.write(f"The required a sample size in terms of **Sensitivity** is:")
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <div style="
            font-size: 36px;
            font-weight: bold;
            background-color: yellow;
            padding: 10px;
            border-radius: 10px;
            text-align: center;">
            {dds1}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"And the required a sample size in terms of **Specificity** is:")
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <div style="
            font-size: 36px;
            font-weight: bold;
            background-color: yellow;
            padding: 10px;
            border-radius: 10px;
            text-align: center;">
            {dds2}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"to achive {Se}% Sensitivity and {Sp}% Specificity with {d}% absolute precision **95%** confidence level, by assuming that {p}% prevalence of the event or factor, where the design effect is **{designEffect}** with **{(drpt)}%** drop-out from the sample.")
    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)

st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **Sensitivity Sample Size Formula**")
st.latex(r"""
n_{Se} = \left( \frac{Z_{1-\alpha/2}}{d \times p} \right)^2 \times Se (1 - Se) \times DE
""")

st.markdown("### **Specificity Sample Size Formula**")
st.latex(r"""
n_{Sp} = \left( \frac{Z_{1-\alpha/2}}{d \times (1-p)} \right)^2 \times Sp (1 - Sp) \times DE
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")
st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{1-alpha /2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( d \)**: Precision (margin of error).
- **\( p \)**: Prevalence of the condition.
- **\( Se \) (Sensitivity)**: Anticipated Sensitivity.
- **\( Sp \) (Specificity)**: Anticipated SPecificity.
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of individuals per cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")

st.subheader("References")

st.markdown("""
1. **Buderer, N. M. F. (1996).** Statistical methodology: I. Incorporating the prevalence of disease into the sample size calculation for sensitivity and specificity. Acadademic Emergency Medicine, 3(9), 895-900. Available at: [https://pubmed.ncbi.nlm.nih.gov/8870764/](https://pubmed.ncbi.nlm.nih.gov/8870764/)
""")

st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]** | Contact: [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")
