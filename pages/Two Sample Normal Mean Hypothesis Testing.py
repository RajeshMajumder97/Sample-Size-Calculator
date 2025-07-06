import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

st.set_page_config(page_title="StydySizer | Two Sample Normal Mean Hypothesis Testing",
                   page_icon="🧮")

st.markdown("""
    <style>
    button[data-testid="stBaseButton-header"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.title("Sample Size Calculation for Two sample Mean Test | H0: Mu1=Mu2")

## Functuion
def nSampleMean(sigma=0.01,Pw=0.8,delta=0.05,Conf=0.95,designEf=1,dropOut=0):
    n= (2*(norm.ppf(1-(1-Conf)/2)+norm.ppf(Pw))**2)*(sigma/delta)**2                               
    return(abs(round((n/(1-dropOut))*designEf)))

# Initialize history store
if "mean_test_history" not in st.session_state:
    st.session_state.mean_test_history = []

sigma = st.sidebar.number_input("Standard Deviation (SD)",value=15.0,min_value=0.01,help= "values in decimal.")
delta = st.sidebar.number_input("Expected difference", value=10.0,min_value=0.0)
power= st.sidebar.number_input("Power (%)", value=80.0,min_value=0.0,max_value=100.0)
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
def make_mean_test_history_label(sigma, delta, power, drpt, designEffect, m=None, ICC=None, method="Given"):
    if method == "Given":
        return f"Sigma={sigma}, difference={delta}, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
    else:
        return (f"Sigma={sigma}, difference={delta}, Power={power}%, DropOut={drpt}%, "
                f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

# Select from history
selected_history = None
selected_label = None

if st.session_state.mean_test_history:
    st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
    mean_test_options = [make_mean_test_history_label(**entry) for entry in st.session_state.mean_test_history]
    selected_label = st.selectbox("Choose a past input set:", mean_test_options, key="mean_test_history_selector")

    if selected_label:
        selected_history = next((item for item in st.session_state.mean_test_history
                                 if make_mean_test_history_label(**item) == selected_label), None)
        hist_submit = st.button("🔁 Recalculate from Selected History")
    else:
        hist_submit = False
else:
    hist_submit = False


if go or hist_submit:
    if hist_submit and selected_history:
        # Use selected history
        sigma= selected_history["sigma"]
        delta= selected_history["delta"]
        power = selected_history["power"]
        drpt = selected_history["drpt"]
        designEffect = selected_history["designEffect"]
    else:
        # Add current input to history
        new_entry = {
            "sigma":sigma,
            "delta":delta,
            "power":power,
            "drpt":drpt,
            "designEffect":designEffect,
            "m":m,
            "ICC":ICC,
            "method":x
        }
        st.session_state.mean_test_history.append(new_entry)

    confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
    out=[]

    for conf in confidenceIntervals:
        sample_size= nSampleMean(sigma=sigma,delta=delta,Pw=(power/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
        out.append(sample_size)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        "Sample Size": out
    })

    dds= nSampleMean(sigma=sigma,delta=delta,Pw=(power/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))

    st.write(f"The study would require a total sample size of:")
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <div style="
            font-size: 36px;
            font-weight: bold;
            background-color: #48D1CC;
            padding: 10px;
            border-radius: 10px;
            text-align: center;">
            {2*dds}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write(f""" number of individuals (i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{dds}</span> individuals in each group) to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, for detecting a true difference in means between the test and the reference group {delta} units, by assuming the standard deviation of the differences to be {sigma} units, where the design effect is **{round(designEffect,1)}** with **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)


st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **Two-Sample Mean Hypothesis Test Sample Size Formula**")

st.latex(r"""
n = \frac{2 (Z_{1-(\alpha/2)} + Z_{1-\beta})^2 \cdot \sigma^2}{\delta^2} \times \frac{DE}{1 - \text{Dropout\%}}
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( Z_{1-beta} \)**: Power.
- **\( sigma \)**: Population standard deviation.
- **\( delta \)**: Expected difference between two means.
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
- **Dropout%**: Anticipated percentage of dropout in the study.
""")

st.subheader("📌 References")

st.markdown("""
1. **Naing, N. N. (2011).** A practical guide on determination of sample size in health sciences research. Kelantan: Pustaka Aman Press.
""")

st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")


st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")