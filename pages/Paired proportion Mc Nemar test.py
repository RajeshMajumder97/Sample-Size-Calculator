import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
import math
st.set_page_config(page_title="Paired proportion Mc Nemar test",
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
st.title("Sample Size Calculation for Paired Proportion test (Mc Nemar's test) | H0: p10=p01")

## Functuion
def nSampleProp(p10=0.5,p01=0.6,Pw=0.8,Conf=0.95,designEf=1,dropOut=0):
    n= ((norm.ppf(1-(1-Conf)/2)*math.sqrt(p01+p10))+(norm.ppf(Pw)*math.sqrt(p10+p01-(p01-p10)**2)))**2/(p10-p01)**2
    return(abs(round((n/(1-dropOut))*designEf)))

p10= st.sidebar.number_input("1st Discordant Pair Proportion (P10 / + to -) (%)",value=50.0,min_value=0.0,max_value=100.0)
p01= st.sidebar.number_input("2nd Discordant Pair Proportion (P01 / - to +) (%)",value=40.0,min_value=0.0,max_value=100.0)
power= st.sidebar.number_input("Power (%)", value=80.0,min_value=0.0,max_value=100.0)
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
        sample_size= nSampleProp(p10=(p10/100),p01=(p01/100),Pw=(power/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
        out.append(sample_size)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        "Sample Size": out
    })

    dds= nSampleProp(p10=(p10/100),p01=(p01/100),Pw=(power/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))
    
    st.write(f"The study would require a sample size of:")
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
    st.write(f"""to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, for detecting a difference of {p10-p01}% between the discordant proportions, by assuming that {p10}% of the pairs switch from positive to negative and {p01}% from negative to positive, where the design effect is **{round(designEffect,1)}** with **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)


st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **Paired Proportion Test (McNemar's Test) Sample Size Formula**")

st.latex(r"""
n = \frac{\left( Z_{1-\alpha/2} \sqrt{p_{01} + p_{10}} + Z_{1-\beta} \sqrt{p_{10} + p_{01} - (p_{01} - p_{10})^2} \right)^2}{(p_{10} - p_{01})^2} \times DE
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( Z_{1-beta} \)**: Power.
- **\( p_{10} \)** & **\( p_{01} \)**:  Discordant pairs respectively.
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")

st.subheader("📌 References")

st.markdown("""
1. **Machin, D., Campbell, M. J., Tan, S. B., & Tan, S. H. (2018).** Sample Size Tables for Clinical Studies (3rd ed.). Wiley-Blackwell.
""")

st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")