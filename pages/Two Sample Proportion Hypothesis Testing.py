import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

st.set_page_config(page_title="Two Sample Proportion Hypothesis Testing",
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
st.title("Sample Size Calculation for Proportion Test | H0: P1=P2")

## Functuion
def nSampleProp(p1=0.5,p2=0.5,delta=0.0,Pw=0.8,Conf=0.95,designEf=1,dropOut=0):
    n= (((norm.ppf(1-(1-Conf)/2)+norm.ppf(Pw))**2)*(p1*(1-p1)+p2*(1-p2)))/(delta**2)
    return(abs(round((n/(1-dropOut))*designEf)))

p1 = st.sidebar.number_input("Proportion in 1st(Reference) Group (%)",value=50.0,min_value=0.0,max_value=100.0)
p2 = st.sidebar.number_input("Proportion in 2nd(Test) Group (%)",value=40.0,min_value=0.0,max_value=100.0)
delta = abs(p2-p1)
power= st.sidebar.number_input("Power (%)", value=80.0,min_value=0.0,max_value=100.0)
drpt= st.sidebar.number_input("Drop-Out (%)",min_value=0.0,value=0.0,max_value=100.0)

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
        sample_size= nSampleProp(p1=(p1/100),p2=(p2/100),delta=(delta/100),Pw=(power/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
        out.append(sample_size)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        "Sample Size": out
    })

    dds= nSampleProp(p1=(p1/100),p2=(p2/100),delta=(delta/100),Pw=(power/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))
    
    st.write(f"The study would require a sample size of:")
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <div style="
            font-size: 36px;
            font-weight: bold;
            background-color: yellow;
            padding: 10px;
            border-radius: 10px;
            text-align: center;">
            {dds}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"for each group(i.e. a total sample size of {2*dds}) to achive a power of {(power)}% and **95%** confidence level, for detecting a difference in proportions of {delta} between the test and the reference group, by assuming that{p1}% of the subjects in the reference population have the factor of interest, where the design effect is **{designEffect}** with **{(drpt)}%** drop-out from the sample.")
    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)


st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **Two-Sample Proportion Hypothesis Test Sample Size Formula**")

st.latex(r"""
n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (p_1(1 - p_1) + p_2(1 - p_2))}{\delta^2} \times DE
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( Z_{1-beta} \)**: Power.
- **\( p_1 \)**, **\( p_2 \)**: Proportions in the first and second groups.
- **\( \delta \)**: Expected difference between the two proportions.
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of individuals per cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")

st.subheader("📌 References")

st.markdown("""
1. **Lemeshow, S., Hosmer Jr., D.W., Klar, J., Lwanga, S.K. (1990).** Adequacy of sample size in health studies. England: John Wiley & Sons.
""")

st.markdown("---")
st.markdown("**Developed by [Your Name]** | Streamlit App for Sample Size Estimation")
