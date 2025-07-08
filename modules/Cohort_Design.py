import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
import math
def main():

    st.title("Sample Size Calculation for Case Control Design: Odds Ratio | H0: OR=1")
    st.markdown(
        """
        <style>
        button[data-testid="stBaseButton-header"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    ## Functuion
    def nSampleRR(p1=0.5,RR=1.0,Pw=0.8,Conf=0.95,designEf=1,dropOut=0):
        p2= p1*RR
        n= ((norm.ppf(1-(1-Conf)/2)+ norm.ppf(Pw))**2/(np.log(RR)**2)) * ((p1*(1-p1)+p2*(1-p2))/(p1-p2)**2)      
        return(abs(round((n/(1-dropOut))*designEf)))

    # Initialize history store
    if "cohort_history" not in st.session_state:
        st.session_state.cohort_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")
    p1 = st.sidebar.number_input("Proportion of disease in unexposed group (control) (%)",value=40.0,min_value=0.0,max_value=99.9)
    R = st.sidebar.number_input("Anticipated Relative Risk (RR)", value=1.45,min_value=0.00001,help= "values in decimal.")

    if R == 1.0:
        st.sidebar.warning("RR = 1.0 means no difference. Please use RR ≠ 1 to detect an effect.")
        st.stop()

    power= st.sidebar.number_input("Power (%)", value=80.00,min_value=50.0,max_value=99.9)
    drpt= st.sidebar.number_input("Drop-Out",value=0.0,min_value=0.0,max_value=50.0)

    x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

    if(x== "Given"):
        designEffect= st.sidebar.number_input("Design Effect", value=1.0,min_value=1.0,max_value=2.0,help= "values in integer. Minimum is 1")
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
    def make_cohort_history_label(p1, R, power, drpt, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"P1={p1}, RR={R}, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            return (f"P1={p1}%, RR={R}, Power={power}%, DropOut={drpt}%, "
                    f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.cohort_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        cohort_options = [make_cohort_history_label(**entry) for entry in st.session_state.cohort_history]
        selected_label = st.selectbox("Choose a past input set:", cohort_options, key="cohort_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.cohort_history
                                    if make_cohort_history_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History")
        else:
            hist_submit = False
    else:
        hist_submit = False


    if go or hist_submit:
        if hist_submit and selected_history:
            # Use selected history
            p1= selected_history["p1"]
            R= selected_history["R"]
            power = selected_history["power"]
            drpt = selected_history["drpt"]
            designEffect = selected_history["designEffect"]
        else:
            # Add current input to history
            new_entry = {
                "p1":p1,
                "R":R,
                "power":power,
                "drpt":drpt,
                "designEffect":designEffect,
                "m":m,
                "ICC":ICC,
                "method":x
            }
            st.session_state.cohort_history.append(new_entry)

        confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
        out=[]

        for conf in confidenceIntervals:
            sample_size= nSampleRR(p1=(p1/100),RR=R,Pw=(power/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
            
            out.append(sample_size)

        df= pd.DataFrame({
            "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
            "Sample Size": out
        })

        dds= nSampleRR(p1=(p1/100),RR=R,Pw=(power/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))

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
        st.write(f"""to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, to expect relative risk or risk ratio as {R}, by assuming that the proportion of disease in unexposed(control) group is {p1}%, where the design effect is **{round(designEffect,1)}** with **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
        st.subheader("List of Sample Sizes at other Confidence Levels")
        st.dataframe(df)


    st.markdown("---")  # Adds a horizontal line for separation

    st.subheader("📌 Formula for Sample Size Calculation")

    st.markdown("### **Cohort Study Sample Size Formula for Relative Risk**")

    st.latex(r"""
    n = \frac{\left( Z_{1-\alpha/2} + Z_{1-\beta} \right)^2}{\ln(RR)^2} \times \frac{p_1(1 - p_1) + p_2(1 - p_2)}{(p_1 - p_2)^2} \times \frac{DE}{1 - \text{Dropout\%}}
    """)

    st.markdown("### **Design Effect Calculation (if clusters are used):**")
    st.latex(r"""
    DE = 1 + (m - 1) \times ICC
    """)

    st.subheader("📌 Description of Parameters")

    st.markdown("""
    - **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
    - **\( Z_{1-beta} \)**: Power.
    - **\( p_1 \)**: Proportion of disease in the **unexposed** (control) group.
    - **\( p_2 \)**: Proportion of disease in the **exposed** group, calculated as: p2= RR*p1
    - **\( RR \)**: Anticipated Relative Risk.
    - **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
    - **\( m \)**: Number of cluster.
    - **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
    - **Dropout%**: Anticipated percentage of dropout in the study.
    """)

    st.subheader("📌 References")

    st.markdown("""
    1. **Kelsey, J. L., Whittemore, A. S., Evans, A. S., & Thompson, W. D. (1996).** Methods in Observational Epidemiology (2nd ed.). Oxford University Press.
    """)

    st.markdown("---")
    st.subheader("Citation")
    st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")


    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")