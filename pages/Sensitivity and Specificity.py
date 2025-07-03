import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
import math

st.set_page_config(page_title="StydySizer | Sensitivity and Specificity",
                   page_icon="🧮")

st.markdown("""
    <style>
    button[data-testid="stBaseButton-header"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


# Streamlit App
st.title("Sample Size Calculation for Sensitivity & Specificity: Estimation")

## Functuion
def nSampleSen(p=0.5,Se=0.80,d=0.05,Conf=0.95,designEf=1,dropOut=0):
    n= ((norm.ppf(1-(1-Conf)/2)/(d*math.sqrt(p)))**2)*(Se*(1-Se))
    return(abs(round((n/(1-dropOut))*designEf)))

def nSampleSpc(p=0.5,Sp=0.70,d=0.05,Conf=0.95,designEf=1,dropOut=0):
    n= ((norm.ppf(1-(1-Conf)/2)/(d*math.sqrt(1-p)))**2)*(Sp*(1-Sp))
    return(round((n/(1-dropOut))*designEf))

# Initialize history store
if "senspe_history" not in st.session_state:
    st.session_state.senspe_history = []


p = st.sidebar.number_input("Prevalence of the Event (%)",value=50.0,min_value=0.0,max_value=100.0)
Se = st.sidebar.number_input("Anticipated Sensitivity (%)",value=80.0,min_value=0.0,max_value=100.0)
Sp = st.sidebar.number_input("Anticipated Specificity (%)",value=70.0,min_value=0.0,max_value=100.0)
d = st.sidebar.number_input("Precision (%)", value=5.0,min_value=0.0,max_value=100.0)
drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=100.0)

x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

if(x== "Given"):
    designEffect= st.sidebar.number_input("Design Effect", value=1.0,min_value=1.0,help= "values in integer. Minimum is 1")
    m=None
    ICC=None
else:
    m= st.sidebar.number_input("Number of cluster",min_value=2)
    ICC= st.sidebar.number_input("ICC",min_value=0.0)
    designEffect= 1+(m-1)*ICC
    col1,col2,col3=st.columns(3)
    col1.metric("Cluster Size (m)",value=m)
    col2.metric("Intra Class Correlation (ICC)",value=ICC)
    col3.metric("Design Effect",value= round(designEffect,2))

# Calculate button
go = st.button("Calculate Sample Size")

# Helper to generate label for dropdown
def make_senspe_history_label(p, Se, Sp, d, drpt, designEffect, m=None, ICC=None, method="Given"):
    if method == "Given":
        return f"Preval={p}%, Senc={Se}%, Spec={Sp}%, Precision={d}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
    else:
        return (f"Preval={p}%, Senc={Se}%, Spec={Sp}%, Precision={d}%, DropOut={drpt}%, "
                f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

# Select from history
selected_history = None
selected_label = None

if st.session_state.senspe_history:
    st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
    senspe_options = [make_senspe_history_label(**entry) for entry in st.session_state.senspe_history]
    selected_label = st.selectbox("Choose a past input set:", senspe_options, key="senspe_history_selector")

    if selected_label:
        selected_history = next((item for item in st.session_state.senspe_history
                                 if make_senspe_history_label(**item) == selected_label), None)
        hist_submit = st.button("🔁 Recalculate from Selected History")
    else:
        hist_submit = False
else:
    hist_submit = False



if go or hist_submit:
    if hist_submit and selected_history:
        # Use selected history
        p= selected_history["p"]
        Se= selected_history["Se"]
        Sp= selected_history["Sp"]
        d= selected_history["d"]
        drpt= selected_history["drpt"]
        designEffect = selected_history["designEffect"]
    else:
        # Add current input to history
        new_entry = {
            "p":p,
            "Se":Se,
            "Sp":Sp,
            "d":d,
            "drpt":drpt,
            "designEffect":designEffect,
            "m":m,
            "ICC":ICC,
            "method":x
        }
        st.session_state.senspe_history.append(new_entry)

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
            background-color: #48D1CC;
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
            background-color: #48D1CC;
            padding: 10px;
            border-radius: 10px;
            text-align: center;">
            {dds2}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"""to achive {Se}% Sensitivity and {Sp}% Specificity with {d}% absolute precision <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, by assuming that {p}% prevalence of the event or factor, where the design effect is **{round(designEffect,1)}** with **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
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
- **\( m \)**: Number of cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")

st.subheader("References")

st.markdown("""
1. **Buderer, N. M. F. (1996).** Statistical methodology: I. Incorporating the prevalence of disease into the sample size calculation for sensitivity and specificity. Acadademic Emergency Medicine, 3(9), 895-900. Available at: [https://pubmed.ncbi.nlm.nih.gov/8870764/](https://pubmed.ncbi.nlm.nih.gov/8870764/)
""")

st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")


st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")