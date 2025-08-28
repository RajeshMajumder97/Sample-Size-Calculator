import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
#from Functions.SSound import *

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
    #st.subheader("🗣️ Know how to calculate sample size.")
    #audio_file = open("Audio/Mean Estimation.mp3", "rb")
    #audio_bytes = audio_file.read()
    #st.audio(audio_bytes, format="audio/mp3")

    #try:
    #    st.markdown("### Know how to use this tool to calculate sample size:")
    #    with open("Audio/Mean Estimation.mp3", "rb") as audio_file:
    #        audio_bytes = audio_file.read()
    #    st.audio(audio_bytes, format="audio/mp3")
    #except FileNotFoundError:
    #    st.warning("Audio file not found. Please make sure 'Audio/Mean Estimation.mp3' exists in the deployed project.")

    # Inject CSS/JS override
    st.markdown(
        """
        <style>
        /* Hide the up/down spin buttons on number_input */
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input[type=number] {
            -moz-appearance: textfield; /* Firefox */
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

    sigma = st.sidebar.number_input("Standard Deviation (SD)",value=15.0,min_value=0.01,format="%.6g",help= "Enter a value >0")
    ads= st.sidebar.radio("Choose Precision Option",options=['Absolute Precision','Relative Precision'],help="This represents how precisely you want to estimate the true mean in the population. A smaller margin of error leads to a larger required sample size and a narrower confidence interval. For instance, suppose a nutritional study finds that the average systolic blood pressure among participants is 130 mmHg. If we use a 5-unit absolute precision, we can say with confidence that the true mean blood pressure lies between 125 mmHg (130−5) and 135 mmHg (130+5). However, if we use a 5% relative precision, the confidence range becomes 123.5 mmHg (130−5% of 130) to 136.5 mmHg (130+5% of 130). The choice between absolute and relative precision affects how narrowly we can define the likely range of the true average in the broader population.")

    if(ads=='Absolute Precision'):
        d = st.sidebar.number_input("Absoulte Precision (d)", value=1.5,min_value=0.00,max_value=100.0,help="Enter an integer value (e.g., 1.5)")
        d1=d
        mu=None
    else:
        d = st.sidebar.number_input("Relative Precision(%)", value=5.0,min_value=0.00,max_value=99.99,help="Enter a percentage value (e.g., 5%)")
        mu= st.sidebar.number_input("Expected Mean", value=35.0,min_value=0.01,max_value=1e6,help="Enter the expected mean value of the outcome. Must be positive (e.g., average blood pressure, weight, score, etc.).")
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
        m= st.sidebar.number_input("Number of Clusters (m)",min_value=2,value=2,help="Enter an integer value (e.g., 4)")
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
                return f"Sigma={sigma},Precision method={absolute}, Precision(abs)={round(d1,2)}, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
            else:
                return f"Sigma={sigma},Precision method={absolute}, Precision(relt({d}%))={round(d1,2)}%, Ant.Mean={mu}, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            if absolute=='Absolute Precision':
                return (f"Sigma={sigma}, Precision(abs)={d1}, DropOut={drpt}%, "
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
        tabs = st.tabs(["Tabulate", "Precision V/s Confidence Table" ,"Visualisation"])
        with tabs[0]:
            if hist_submit and selected_history:
                # Use selected history
                sigma= selected_history["sigma"]
                d= selected_history["d"]
                mu= selected_history["mu"]
                d1= selected_history["d1"]
                drpt = selected_history["drpt"]
                ads= selected_history["absolute"]
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
                sample_size= nSampleMean(sigma=sigma,d=d1,Conf=conf,designEf=designEffect,dropOut=(drpt/100))
                
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

        with tabs[1]:
            if(ads=='Absolute Precision'):
                st.markdown("### For Absolute Precision, no cross table is available.")
                cross_table=None
            else:
                # D efine power and confidence levels
                precision = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
                conf_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

                st.subheader("📈 Sample Size Cross Table for Different Precisions and Confidence Levels")

                power_labels = [f"{int(p * 100)}%" for p in precision]
                conf_labels = [f"{int(c * 100)}%" for c in conf_levels]
                cross_table = pd.DataFrame(index=conf_labels, columns=power_labels)
                # Fill the cross table
                for i, conf in enumerate(conf_levels):
                    for j, d_val in enumerate(precision):
                        d1= (d_val*mu)
                        ss = nSampleMean(sigma=sigma,d=d1,Conf=conf,designEf=designEffect,dropOut=(drpt/100))
                        cross_table.iloc[i, j] = ss
                # Label table
                cross_table.index.name = "Confidence levels (%)"
                cross_table.columns.name = "Precision(%)"
                st.dataframe(cross_table)
                st.write("**Rows are Confidence levels; Columns are Precision**")
                #st.session_state["cross_table"] = cross_table
        with tabs[2]:
            if(ads=='Absolute Precision'):
                st.markdown("### For Absolute Precision, no visualization is available.")
            else:
                import matplotlib.pyplot as plt

                # Prepare precision and confidence values from cross_table
                precision = [int(col.strip('%')) for col in cross_table.columns]
                conf_levels = [int(row.strip('%')) for row in cross_table.index]

                precision_sorted = sorted(precision)
                conf_levels_sorted = sorted(conf_levels)

                power_labels = [f"{p}%" for p in precision_sorted]
                conf_labels = [f"{cl}%" for cl in conf_levels_sorted]

                col1, col2 = st.columns(2)

                # === Plot 1: Sample Size vs Precision at fixed Confidence Levels ===
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    conf_levels_to_plot = [80, 95, 97, 99]
                    for cl in conf_levels_to_plot:
                        cl_label = f"{cl}%"
                        if cl_label in cross_table.index:
                            sample_sizes = cross_table.loc[cl_label, power_labels].astype(float).tolist()
                            ax1.plot(precision_sorted, sample_sizes, marker='o', linestyle='-', label=f'CL {cl_label}')
                    ax1.set_title("Sample Size vs Precision")
                    ax1.set_xlabel("Precision (%)")
                    ax1.set_ylabel("Sample Size")
                    ax1.grid(True)
                    ax1.legend(title="Confidence Level")
                    st.pyplot(fig1)

                # === Plot 2: Sample Size vs Confidence Level at fixed Precision Levels ===
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    precision_levels_to_plot = [2, 3, 5, 10]
                    for pr in precision_levels_to_plot:
                        pr_label = f"{pr}%"
                        if pr_label in cross_table.columns:
                            sample_sizes = cross_table[pr_label].astype(float).tolist()
                            ax2.plot(conf_levels_sorted, sample_sizes, marker='s', linestyle='--', label=f'Precision {pr_label}')
                    ax2.set_title("Sample Size vs Confidence Level")
                    ax2.set_xlabel("Confidence Level (%)")
                    ax2.set_ylabel("Sample Size")
                    ax2.grid(True)
                    ax2.legend(title="Precision")
                    st.pyplot(fig2)

                st.markdown("---")
                with st.expander("💡Show the Interpretation of the plots"):
                    st.markdown("### Plot 1: Sample Size vs Precision")
                    st.markdown("- As **precision becomes tighter (i.e., smaller %) the required sample size increases** exponentially.")
                    st.markdown("- Higher confidence levels (e.g., 99%) require larger sample sizes than lower ones (e.g., 80%) for the same precision.")
                    st.markdown("### Plot 2: Sample Size vs Confidence Level")
                    st.markdown("- As **confidence level increases**, so does the **required sample size** to ensure the estimate remains within the desired precision.")
                    st.markdown("- At lower confidence (e.g., 70–80%), sample size requirements are modest, but they grow rapidly beyond 95%, especially at tighter precision levels.")

    st.markdown("---")  # Adds a horizontal line for separation
    with st.expander("Show the formula and the references"):
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
        1. Chow, S. C., Shao, J., & Wang, H. (2008). Sample Size Calculations in Clinical Research. 2nd ed. Chapman & Hall/CRC.
        2. Hulley, S. B., Cummings, S. R., et al. (2013). Designing Clinical Research. 4th ed. Lippincott Williams & Wilkins.
        """)

    st.markdown("---")
    st.subheader("Citation")
    from datetime import datetime
    # Get current date and time
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    # Citation with access timestamp
    st.markdown(f"""
    *Majumder, R. (2025). StudySizer: A sample size calculator (Version 0.1.0). Available online: [https://studysizer.streamlit.app/](https://studysizer.streamlit.app/). Accessed on {now}.*
    """)


    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")
