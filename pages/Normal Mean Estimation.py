import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

st.set_page_config(page_title="Normal Mean Estimation",
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
st.title("Sample Size Calculation for Mean: Mean Estimation")

## Functuion
def nSampleMean(sigma=0.01,d=0.05,Conf=0.95,designEf=1,dropOut=0):
    n= ((norm.ppf(1-((1-Conf)/2))/d)**2)*(sigma**2)
    return(abs(round((n/(1-dropOut))*designEf)))

sigma = st.sidebar.number_input("Standard Deviation (SD)",value=15.0,min_value=0.01,help= "values in decimal.")
ads= st.sidebar.radio("Choose Precision Option",options=['Absolute Precision','Relative to the Proportion'])

if(ads=='Absolute Precision'):
    d = st.sidebar.number_input("Absoulte Precision", value=5.0,min_value=0.00,max_value=100.0)
    d1=d
else:
    d = st.sidebar.number_input("Relative Precision(%)", value=5.0,min_value=0.00,max_value=100.0)
    mu= st.sidebar.number_input("Anticipated Mean", value=35.0,min_value=0.00,max_value=100.0)
    d1= (d/100)*mu
    col1,col2,col3=st.columns(3)
    col1.metric("Relative Precision(%)",value=d)
    col2.metric("Anticipated Mean",value=mu)
    col3.metric("Precision",value= round(d1,2))


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
        sample_size= nSampleMean(sigma=sigma,d=1,Conf=conf,designEf=designEffect,dropOut=(drpt/100))
        
        out.append(sample_size)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        "Sample Size": out
    })
    dds= nSampleMean(sigma=sigma,d=d1,Conf=0.95,designEf=designEffect,dropOut=(drpt/100))
    
    if(ads=='Absolute Precision'):
        st.write(f"Assuming a normal distribution with a standard deviation of **{sigma}**,the study would require a sample size of:")
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
        st.write(f"for estimating mean with absolute precision **{(d)}** and **95%** confidence level, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.")
    else:
        st.write(f"Assuming a normal distribution with a standard deviation of **{sigma}**,the study would require a sample size of:")
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
        st.write(f"for estimating mean with relative precision **({mu}*{d}%= ) {round(d,1)}** and **95%** confidence level, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.")


    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)



st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **Sample Size Formula for Mean Estimation**")
st.latex(r"""
n = \left( \frac{Z_{1-\alpha/2} \cdot \sigma}{d} \right)^2 \times DE
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{1-\alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( \sigma \)**: Population standard deviation.
- **\( d \)**: Absolute Precision (margin of error).
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of individuals per cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")

st.subheader("📌 References")

st.markdown("""
1. **Naing, N. N. (2003).** Determination of Sample Size.The Malaysian Journal of Medical Sciences: MJMS,10(2), 84-86. Available at: [https://pubmed.ncbi.nlm.nih.gov/23386802/](https://pubmed.ncbi.nlm.nih.gov/23386802/)
""")

st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")