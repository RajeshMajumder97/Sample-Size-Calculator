import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
import math

def main():


    # Streamlit App
    st.title("Sample Size Calculation for Case Control Design: Odds Ratio | H0: OR=1")

    ## Functuion
    def nSampleOR(p2=0.5,OR=1.0,Pw=0.8,Conf=0.95,designEf=1,dropOut=0):
        n= ((norm.ppf(1-(1-Conf)/2)+ norm.ppf(Pw))**2/(np.log(OR)**2)) * (1/(p2*(1-p2)))       
        return(abs(round((n/(1-dropOut))*designEf)))

    # Initialize history store
    if "case_control_history" not in st.session_state:
        st.session_state.case_control_history = []
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")
    #p2 = st.sidebar.number_input("Proportion of Exposed among controls (%)",value=40.0,min_value=0.0,max_value=100.0)
    #R = st.sidebar.number_input("Anticipated odds ratio (OR)", value=1.0,min_value=0.00001,help= "values in decimal.")
    #power= st.sidebar.number_input("Power (%)", value=80.0,min_value=0.0,max_value=100.0)
    #drpt= st.sidebar.number_input("Drop-Out",value=0.0,max_value=1.0,help="value in decimal")

    p2 = st.sidebar.number_input("Proportion of Exposed among controls (%)",value=40.0,min_value=0.0,max_value=99.9)
    R = st.sidebar.number_input("Anticipated odds ratio (OR)", value=1.5,min_value=0.00001,help= "values in decimal.")

    if R == 1.0:
        st.sidebar.warning("OR should not be 1.0 under H₁. Change it to reflect a real difference.")
        st.stop()

    power= st.sidebar.number_input("Power (%)", value=80.00,min_value=50.0,max_value=99.9)
    drpt= st.sidebar.number_input("Drop-Out",value=0.0,min_value=0.0,max_value=50.0)


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
    def make_case_control_history_label(p2, R, power, drpt, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"P1={p2}, OR={R}, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            return (f"P1={p2}%, OR={R}, Power={power}%, DropOut={drpt}%, "
                    f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.case_control_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        case_control_options = [make_case_control_history_label(**entry) for entry in st.session_state.case_control_history]
        selected_label = st.selectbox("Choose a past input set:", case_control_options, key="case_control_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.case_control_history
                                    if make_case_control_history_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History")
        else:
            hist_submit = False
    else:
        hist_submit = False


    if go or hist_submit:
        if hist_submit and selected_history:
            # Use selected history
            p2= selected_history["p2"]
            R= selected_history["R"]
            power = selected_history["power"]
            drpt = selected_history["drpt"]
            designEffect = selected_history["designEffect"]
        else:
            # Add current input to history
            new_entry = {
                "p2":p2,
                "R":R,
                "power":power,
                "drpt":drpt,
                "designEffect":designEffect,
                "m":m,
                "ICC":ICC,
                "method":x
            }
            st.session_state.case_control_history.append(new_entry)







        confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
        out=[]

        for conf in confidenceIntervals:
            sample_size= nSampleOR(p2=(p2/100),OR=R,Pw=(power/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
            out.append(sample_size)

        df= pd.DataFrame({
            "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
            "Sample Size": out
        })

        dds= nSampleOR(p2=(p2/100),OR=R,Pw=(power/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))

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
        st.write(f"""to achive a power of {round(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, to expect odds ratio as {R}, by assuming that the proportion of exposed among controls is {p2}%, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
        st.subheader("List of Sample Sizes at other Confidence Levels")
        st.dataframe(df)

    st.markdown("---")  # Adds a horizontal line for separation

    st.subheader("📌 Formula for Sample Size Calculation")

    st.markdown("### **Case-Control Study Sample Size Formula for Odds Ratio**")

    st.latex(r"""
    n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2}{\ln(OR)^2} \times \left( \frac{1}{p_0 (1 - p_0)} \right) \times \frac{DE}{1 - \text{Dropout\%}}
    """)

    st.markdown("### **Design Effect Calculation (if clusters are used):**")
    st.latex(r"""
    DE = 1 + (m - 1) \times ICC
    """)

    st.subheader("📌 Description of Parameters")

    st.markdown("""
    - **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
    - **\( Z_{1-beta} \)**: Power.
    - **\( p_0 \)**: Proportion of exposed individuals in the control group.
    - **\( OR \)**: Anticipated Odds Ratio.
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