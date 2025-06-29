import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

st.set_page_config(page_title="Intraclass Correlation",
                   page_icon="🧊")



# Streamlit App
st.title("Sample Size Calculation for Intraclass Correlation")

## Functuion
def nSampleICC(n=5,rho0=2,rho1=0.8,Conf=0.95,Pw=0.8,designEf=1,dropOut=0):
    Z_alpha = norm.ppf(1 - (1-Conf) / 2)  
    Z_beta = norm.ppf(Pw)           
    numerator = 1 + (2 * (Z_alpha + Z_beta) ** 2 * n)
    denominator = (np.log((1 + (n * rho0 / (1 - rho0))) / (1 + (n * rho1 / (1 - rho1))))) ** 2 * (n - 1)

    N = numerator / denominator
    return(abs(round((N/(1-dropOut))*designEf)))

# Initialize history store
if "icc_history" not in st.session_state:
    st.session_state.icc_history = []

Obj = st.sidebar.number_input("Observation/Subject (n)",value=5,min_value=0,help= "values in integer")
st.sidebar.text("Number of repeted observatiuons\n by different judges\n per subject,replicates")
power= st.sidebar.number_input("Power (%)",value=80.0,min_value=50.0,max_value=99.9)
minAR= st.sidebar.number_input("Minimum acceptable reliability (rho_0) (%)",value=60.0,min_value=0.0,max_value=100.0)
st.sidebar.text("The lowest limit of reliability\n you would accept")
ERR= st.sidebar.number_input("Expected reliability (rho_1) (%)",value=80.0,min_value=0.0,max_value=100.0)
st.sidebar.text("The level of reliability\n you can expect from the study")
drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=50.0)

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
def make_icc_history_label(Obj, minAR, ERR, power, drpt, designEffect, m=None, ICC=None, method="Given"):
    if method == "Given":
        return f"Subject={Obj}, Power={power}%, rho_0={minAR}%, rho_1={ERR}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
    else:
        return (f"Subject={Obj}%, Power={power}%, rho_0={minAR}%, rho_1={ERR}%, DropOut={drpt}%, "
                f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

# Select from history
selected_history = None
selected_label = None

if st.session_state.icc_history:
    st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
    icc_options = [make_icc_history_label(**entry) for entry in st.session_state.icc_history]
    selected_label = st.selectbox("Choose a past input set:", icc_options, key="icc_history_selector")

    if selected_label:
        selected_history = next((item for item in st.session_state.icc_history
                                 if make_icc_history_label(**item) == selected_label), None)
        hist_submit = st.button("🔁 Recalculate from Selected History")
    else:
        hist_submit = False
else:
    hist_submit = False


if go or hist_submit:
    if hist_submit and selected_history:
        # Use selected history
        Obj= selected_history["Obj"]
        minAR= selected_history["minAR"]
        ERR= selected_history["ERR"]
        power = selected_history["power"]
        drpt = selected_history["drpt"]
        designEffect = selected_history["designEffect"]
    else:
        # Add current input to history
        new_entry = {
            "Obj":Obj,
            "minAR":minAR,
            "ERR":ERR,
            "power":power,
            "drpt":drpt,
            "designEffect":designEffect,
            "m":m,
            "ICC":ICC,
            "method":x
        }
        st.session_state.icc_history.append(new_entry)

    confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
    out=[]

    for conf in confidenceIntervals:
        sample_size= nSampleICC(n=Obj,rho0=(minAR/100),rho1=(ERR/100),Conf=conf,Pw=(power/100),designEf=designEffect,dropOut=(drpt/100))
        out.append(sample_size)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        "Sample Size": out
    })

    dds= nSampleICC(n=Obj,rho0=(minAR/100),rho1=(ERR/100),Conf=0.95,Pw=(power/100),designEf=designEffect,dropOut=(drpt/100))

    st.write(f"The reliability study design would require a sample size of:")
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
    st.write(f"""for the estimation of Intraclass Correlation to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, by assuming that {Obj} number of repeated observations per subject by different judges with {minAR}% minimum acceptable reliability while the expected reliability is {ERR}%, where the design effect is **{round(designEffect,1)}** with **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)

st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Mathematical Formula for Sample Size Calculation")

st.markdown("### **Sample Size Formula for Intraclass Correlation (ICC) Estimation**")

st.latex(r"""
N = \frac{1 + 2(Z_{\alpha} + Z_{\beta})^2 \cdot n}{\left(\ln\left(\frac{1 + \frac{n \rho_0}{1 - \rho_0}}{1 + \frac{n \rho_1}{1 - \rho_1}}\right)\right)^2 (n - 1)} \times DE
""")

st.markdown("### **Design Effect Calculation (if clusters are used):**")
st.latex(r"""
DE = 1 + (m - 1) \times ICC
""")

st.subheader("📌 Description of Parameters")

st.markdown("""
- **\( Z_{alpha} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
- **\( Z_{beta} \)**: Standard normal quantile for power (1 - beta).
- **\( n \)**: Number of repeated observations per subject.
- **\( rho_0 \)**: Minimum acceptable reliability (null hypothesis ICC).
- **\( rho_1 \)**: Expected reliability (alternative hypothesis ICC).
- **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
- **\( m \)**: Number of cluster.
- **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
""")

st.subheader("📌 References")

st.markdown("""
1. **Walter, S.D., Eliasziw, M., Donner, A. (1998).** Sample size and optimal designs for reliability studies. Statistics in medicine, 17, 101-110. Available at: [https://pubmed.ncbi.nlm.nih.gov/9463853/](https://pubmed.ncbi.nlm.nih.gov/9463853/)
""")

st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")


st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")