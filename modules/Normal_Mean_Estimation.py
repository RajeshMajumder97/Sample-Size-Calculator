import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

def main():

    #    st.set_page_config(page_title="StydySizer | Normal Mean Estimation",
    #                    page_icon="🧮")
    #
    st.title("Sample Size Calculation for Mean: Mean Estimation")
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
    def nSampleMean(sigma=0.01,d=0.05,Conf=0.95,designEf=1,dropOut=0):
        n= ((norm.ppf(1-((1-Conf)/2))/d)**2)*(sigma**2)
        return(abs(round((n/(1-dropOut))*designEf)))

    # Initialize history store
    if "mean_history" not in st.session_state:
        st.session_state.mean_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")

    sigma = st.sidebar.number_input("Standard Deviation (SD)",value=15.0,min_value=0.01,help= "Enter a value >0")
    ads= st.sidebar.radio("Choose Precision Option",options=['Absolute Precision','Relative to the Proportion'])

    if(ads=='Absolute Precision'):
        d = st.sidebar.number_input("Absoulte Precision (d)", value=5.0,min_value=0.00,max_value=100.0,help="Enter an integer value (e.g., 5)")
        d1=d
        mu=None
    else:
        d = st.sidebar.number_input("Relative Precision(%)", value=5.0,min_value=0.00,max_value=99.99,help="Enter a percentage value (e.g., 5%)")
        mu= st.sidebar.number_input("Anticipated Mean", value=35.0,min_value=0.01,max_value=1e6,help="Enter the expected mean value of the outcome. Must be positive (e.g., average blood pressure, weight, score, etc.).")
        d1= (d/100)*mu
        col1,col2,col3=st.columns(3)
        col1.metric("Relative Precision(%)",value=d)
        col2.metric("Anticipated Mean",value=mu)
        col3.metric("Precision",value= round(d1,2))

    if d1 == 0:
        st.error("Precision cannot be zero.")
        st.stop()

    drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=50.0,help="Enter a percentage value (e.g., 1%)")

    x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

    if(x== "Given"):
        designEffect= st.sidebar.number_input("Design Effect (Given)", value=1.0,min_value=1.0,help= "Enter a decimal value (e.g., 1.5)")
        m=None
        ICC=None
    else:
        m= st.sidebar.number_input("Number of cClusters (m)",min_value=2,value=2,help="Enter an integer value (e.g., 4)")
        ICC= st.sidebar.number_input("Intra-class Correlation (ICC) for Cstering",min_value=0.0,max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
        designEffect= 1+(m-1)*ICC
        col1,col2,col3=st.columns(3)
        col1.metric("Cluster Size (m)",value=m)
        col2.metric("Intra Class Correlation (ICC)",value=ICC)
        col3.metric("Design Effect",value= round(designEffect,2))

    # Calculate button
    go = st.button("Calculate Sample Size")

    # Helper to generate label for dropdown
    def make_mean_history_label(sigma,d, d1, drpt, designEffect,mu=None, m=None, ICC=None, method="Given",absolute='Absolute Precision'):
        if method == "Given":
            if absolute=='Absolute Precision':
                return f"Sigma={sigma}, Precision(abs)={round(d1,2)}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
            else:
                return f"Sigma={sigma}, Precision(relt({d}%))={round(d1,2)}%, Ant.Mean={mu}, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            if absolute=='Absolute Precision':
                return (f"Sigma={sigma}, Precision(abs)={d1}%, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")
            else:
                return (f"Sigma={sigma}, Precision(relt({d}%))={d1}%, Ant.Mean={mu}, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")            

    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.mean_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        mean_options = [make_mean_history_label(**entry) for entry in st.session_state.mean_history]
        selected_label = st.selectbox("Choose a past input set:", mean_options, key="mean_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.mean_history
                                    if make_mean_history_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History")
        else:
            hist_submit = False
    else:
        hist_submit = False


    if go or hist_submit:
        if hist_submit and selected_history:
            # Use selected history
            sigma= selected_history["sigma"]
            d1= selected_history["d1"]
            drpt = selected_history["drpt"]
            designEffect = selected_history["designEffect"]
        else:
            # Add current input to history
            new_entry = {
                "sigma":sigma,
                "d1":d1,
                "drpt":drpt,
                "designEffect":designEffect,
                "m":m,
                "ICC":ICC,
                "method":x,
                "absolute": ads,
                "mu":mu,
                "d":d
            }
            st.session_state.mean_history.append(new_entry)

        confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
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
            st.write(f"""for estimating mean with absolute precision **{(d)}** and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
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
            st.write(f"""for estimating mean with relative precision **({mu}*{d}%= ) {round(d1,1)}** and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)


        st.subheader("List of Sample Sizes at other Confidence Levels")
        st.dataframe(df)


    st.markdown("---")  # Adds a horizontal line for separation

    st.subheader("📌 Formula for Sample Size Calculation")

    st.markdown("### **Sample Size Formula for Mean Estimation**")
    st.latex(r"""
    n = \left( \frac{Z_{1-\alpha/2} \cdot \sigma}{d} \right)^2 \times \frac{DE}{1 - \text{Dropout\%}}
    """)

    st.markdown("### **Design Effect Calculation (if clusters are used):**")
    st.latex(r"""
    DE = 1 + (m - 1) \times ICC
    """)

    st.subheader("📌 Description of Parameters")

    st.markdown("""
    - **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
    - **\( \sigma \)**: Population standard deviation.
    - **\( d \)**: Absolute Precision (margin of error).
    - **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
    - **\( m \)**: Number of cluster.
    - **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
    - **Dropout%**: Anticipated percentage of dropout in the study.
    """)

    st.subheader("📌 References")

    st.markdown("""
    1. **Naing, N. N. (2003).** Determination of Sample Size.The Malaysian Journal of Medical Sciences: MJMS,10(2), 84-86. Available at: [https://pubmed.ncbi.nlm.nih.gov/23386802/](https://pubmed.ncbi.nlm.nih.gov/23386802/)
    """)

    st.markdown("---")
    st.subheader("Citation")
    st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")


    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")