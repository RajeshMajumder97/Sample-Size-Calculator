import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

def main():
    #    st.set_page_config(page_title="StydySizer | Skewed Normal Mean Estimation",
    #                    page_icon="🧮")
    #
    def psn(x, delta):
        """ CDF of the skew-normal distribution """
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    def nsampleSN(cv=0.45, prec=0.05, conf=0.95, nmin=25, nmax=1000, nby=5, nf=15,designeffect=1,dropOut=0):
        f = prec / cv
        g = min(0.99, 2 * cv)  # Gamma
        
        d1 = 0.5 * np.pi * g ** (2 / 3)
        d2 = g * (2 / 3) + (0.5 * (4 - np.pi)) * (2 / 3)
        delta = np.sqrt(d1 / d2)
        
        # Ensure valid range for sqrt
        denominator = np.sqrt(max(1 - delta ** 2, 1e-10))
        lambda_ = delta / denominator
        
        f_hat = None
        N = nmax
        nset = np.arange(nmin, nmax + 1, nby)
        
        for n in nset:
            delta_star = lambda_ / np.sqrt(1 + n * lambda_ ** 2)
            
            f_grid = np.linspace(0, f, nf)
            ff = np.array(np.meshgrid(-f_grid, f_grid)).T.reshape(-1, 2)[1:, :]
            
            for i in range(len(ff)):
                L = np.sqrt(n) * (ff[i, 0] * np.sqrt(1 - 2 * delta_star ** 2 / np.pi) + delta_star * np.sqrt(2 / np.pi))
                U = np.sqrt(n) * (ff[i, 1] * np.sqrt(1 - 2 * delta_star ** 2 / np.pi) + delta_star * np.sqrt(2 / np.pi))
                
                if psn(U, delta_star) - psn(L, delta_star) >= conf:
                    f_hat = ff[i, :]
                    break
            
            if f_hat is not None:
                N = n
                break
        
        return (abs(round((N/(1-dropOut))*designeffect)))

    # Streamlit App
    st.title("Sample Size Calculation for Skew Normal Distribution: Mean Estimation")
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
    # Initialize history store
    if "Sknoormal_history" not in st.session_state:
        st.session_state.Sknoormal_history = []

    st.sidebar.markdown("---")

    st.sidebar.header("🔧 Input Parameters")

    cv = st.sidebar.number_input("Coefficient of Variation (%)",max_value=100.0,value=5.00,min_value=1.00,help="Enter a percentage value (e.g., 5%)")
    prec = st.sidebar.number_input("Precision (%)",value=10.00,min_value=0.00,max_value=100.00,help="Enter a percentage value (e.g., 10%)")
    #conf = st.sidebar.number_input("Confidence Level", max_value=0.99,value=0.95,help= "values in decimal")
    #nmax = st.sidebar.number_input("Maximum Sample Size", value=5000)
    #nmin = st.sidebar.number_input("Minumum Sample Size", value=25,min_value=25)
    drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=50.0,help="Enter a percentage value (e.g., 1%)") 
    x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

    if(x== "Given"):
        designEffect= st.sidebar.number_input("Design Effect (Given)", value=1.0,min_value=1.0,help= "Enter an decimal value (e.g., 1.5)")
        m=None
        ICC=None
    else:
        m= st.sidebar.number_input("Number of Clusters (m)",min_value=2,value=4,help="Enter an integer value (e.g., 4)")
        ICC= st.sidebar.number_input("Intra-class Correlation (ICC) for clustering",min_value=0.0,max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
        designEffect= 1+(m-1)*ICC
        col1,col2,col3=st.columns(3)
        col1.metric("Cluster Size (m)",value=m)
        col2.metric("Intra Class Correlation (ICC)",value=ICC)
        col3.metric("Design Effect",value= round(designEffect,2))

    # Calculate button
    go = st.button("Calculate Sample Size")

    # Helper to generate label for dropdown
    def make_Sknoormal_history_label(cv, prec, drpt, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"CV={cv}%, Precision={prec}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            return (f"CV={cv}%, Precision={prec}%, DropOut={drpt}%, "
                    f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.Sknoormal_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        Sknoormal_options = [make_Sknoormal_history_label(**entry) for entry in st.session_state.Sknoormal_history]
        selected_label = st.selectbox("Choose a past input set:", Sknoormal_options, key="Sknoormal_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.Sknoormal_history
                                    if make_Sknoormal_history_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History")
        else:
            hist_submit = False
    else:
        hist_submit = False


    if go or hist_submit:
        if hist_submit and selected_history:
            # Use selected history
            cv= selected_history["cv"]
            prec= selected_history["prec"]
            drpt= selected_history["drpt"]
            designEffect = selected_history["designEffect"]
        else:
            # Add current input to history
            new_entry = {
                "cv":cv,
                "prec":prec,
                "drpt":drpt,
                "designEffect":designEffect,
                "m":m,
                "ICC":ICC,
                "method":x
            }
            st.session_state.Sknoormal_history.append(new_entry)

        confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
        out=[]
        for conf in confidenceIntervals:
            sample_size= nsampleSN(cv=(cv/100), prec=(prec/100), conf=conf, nmax=10000,nmin=25,designeffect=designEffect,dropOut=(drpt/100))
            out.append(sample_size)
        df= pd.DataFrame({
            "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
            "Sample Size": out
        })
        dds= nsampleSN(cv=(cv/100), prec=(prec/100), conf=0.95, nmax=3000,nmin=25,designeffect=designEffect,dropOut=(drpt/100))
        st.write(f"Asuming that with **{(cv)}%** coefficient of variation in a skewed normal distribution,the study would require a sample size of:")
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
        st.write(f"""for estimating mean with **{(prec)}%** relative precision and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level,where the design effect is **{round(designEffect,1)}** with **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
        st.subheader("List of Sample Sizes at other Confidence Levels")
        st.dataframe(df)
        


    st.markdown("---")  # Adds a horizontal line for separation
    with st.expander("Show the formula and the references"):
        st.subheader("📌 Formula for Sample Size Calculation")
        st.markdown("""
        The **minimum required sample size** for estimating a mean is given by:
        """)

        st.latex(r"""
        n = \left( \frac{Z_{1-\alpha/2}}{f} \right)^2 
        """)
        st.markdown("Where:")
        st.latex(r"""
        f= \frac{ d \bar{x} }{s} =\frac{d}{CV}
                """)

        st.markdown("""
        Where:
        - $d$ isrelative precision of the estimate.
        - $CV$ is the coefficient of variation
        """)

        st.markdown("---")

        st.subheader("References")

        st.markdown("""
        **Trafimow D, Wang T, Wang C.** From a Sampling Precision Perspective, Skewness Is a Friend and Not an Enemy! Educ Psychol Meas. 2019 Feb;79(1):129-150. doi: 10.1177/0013164418764801. Epub 2018 Apr 3. PMID: 30636785; PMCID: PMC6318746.
        """)


    #st.markdown("""
    #    <div style="
    #        background-color: #f9f871;
    #        padding: 10px;
    #        border-left: 5px solid orange;
    #        border-radius: 5px;
    #        font-size: 18px;">
    #        <b>Note:</b> The design effect option is only applicable when doing cluster random sampling, other wise the default is 1 and it is recommended to be done in consultation with a statistician.   
    #    </div>
    #    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Citation")
    from datetime import datetime
    # Get current date and time
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    # Citation with access timestamp
    st.markdown(f"""
    *Majumder, R. (2025). StudySizer: A sample size calculator (Version 0.1.0). Available online: [https://studysizer.netlify.app/](https://studysizer.netlify.app/). Accessed on {now}. [https://doi.org/10.5281/zenodo.16375937](https://doi.org/10.5281/zenodo.16375937).*
    """)


    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")