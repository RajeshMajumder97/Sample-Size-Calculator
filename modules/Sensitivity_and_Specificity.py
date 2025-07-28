import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
import math

def main():
    #    st.set_page_config(page_title="StydySizer | Sensitivity and Specificity",
    #                    page_icon="🧮")
    #
    # Streamlit App
    st.title("Sample Size Calculation for Sensitivity & Specificity: Estimation")

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
    def nSampleSen(p=0.5,Se=0.80,d=0.05,Conf=0.95,designEf=1,dropOut=0):
        n= ((norm.ppf(1-(1-Conf)/2)/(d*math.sqrt(p)))**2)*(Se*(1-Se))
        return(abs(round((n/(1-dropOut))*designEf)))

    def nSampleSpc(p=0.5,Sp=0.70,d=0.05,Conf=0.95,designEf=1,dropOut=0):
        n= ((norm.ppf(1-(1-Conf)/2)/(d*math.sqrt(1-p)))**2)*(Sp*(1-Sp))
        return(round((n/(1-dropOut))*designEf))

    # Initialize history store
    if "senspe_history" not in st.session_state:
        st.session_state.senspe_history = []

    st.sidebar.markdown("---")

    st.sidebar.header("🔧 Input Parameters")

    p = st.sidebar.number_input("Prevalence of the Event (%)",value=50.0,min_value=0.0,max_value=99.99,help="Enter a percentage value (e.g., 50%)")
    Se = st.sidebar.number_input("Expected Sensitivity (%)",value=70.0,min_value=0.0,max_value=99.99,help="Enter a percentage value (e.g., 70%)")
    Sp = st.sidebar.number_input("Expected Specificity (%)",value=60.0,min_value=0.0,max_value=99.99,help="Enter a percentage value (e.g., 60%)")
    d = st.sidebar.number_input("Absolute Precision (%)", value=5.0,min_value=0.0,max_value=50.0,help="Enter a percentage value (e.g., 5%)")
    drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=50.0,help="Enter a percentage value (e.g., 1%)")

    x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

    if(x== "Given"):
        designEffect= st.sidebar.number_input("Design Effect (Given)", value=1.0,min_value=1.0,help= "Enter an decimal value (e.g., 1.5)")
        m=None
        ICC=None
    else:
        m= st.sidebar.number_input("Number of Clusters (m)",min_value=2,value=4,help="Enter an integer value (e.g., 4)")
        ICC= st.sidebar.number_input("Intra-class Correlation (ICC) for clustering",min_value=0.0,max_value=1.0,value=0.05, help="Enter a decimal value (e.g., 0.05)")
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
        tabs = st.tabs(["Tabulate", "Precision V/s Confidence Table" ,"Visualisation"])
        with tabs[0]:
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

            confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
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
        
        with tabs[1]:
            # D efine power and confidence levels
            precision = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
            conf_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

            st.subheader("📈 Sample Size Cross Table for Different Powers and Confidence Levels")

            power_labels = [f"{int(p * 100)}%" for p in precision]
            conf_labels = [f"{int(c * 100)}%" for c in conf_levels]
            cross_table_se = pd.DataFrame(index=conf_labels, columns=power_labels)
            cross_table_sp = pd.DataFrame(index=conf_labels, columns=power_labels)
            # Fill the cross table
            for i, conf in enumerate(conf_levels):
                for j, d_val in enumerate(precision):
                    ss_se =  nSampleSen(p=(p/100),Se=(Se/100),d=d_val,Conf=conf,designEf=designEffect,dropOut=(drpt/100))
                    ss_sp =  nSampleSpc(p=(p/100),Sp=(Sp/100),d=d_val,Conf=conf,designEf=designEffect,dropOut=(drpt/100))
                    
                    cross_table_se.iloc[i, j] = ss_se
                    cross_table_sp.iloc[i, j] = ss_sp
            # Label table
            cross_table_se.index.name = "Confidence levels (%)"
            cross_table_se.columns.name = "Precision(%)"
            cross_table_sp.index.name = "Confidence levels (%)"
            cross_table_sp.columns.name = "Precision(%)"   
            st.subheader("For Sensitivity:")             
            st.dataframe(cross_table_se)
            st.subheader("For Specificity:")             
            st.dataframe(cross_table_sp)                

            st.write("**Rows are Confidence levels; Columns are Precision**")
            #st.session_state["cross_table"] = cross_table
        with tabs[2]:
            import matplotlib.pyplot as plt

            # Prepare precision and confidence values from cross_table
            precision_se = [int(col.strip('%')) for col in cross_table_se.columns]
            conf_levels_se = [int(row.strip('%')) for row in cross_table_se.index]
            precision_sp = [int(col.strip('%')) for col in cross_table_sp.columns]
            conf_levels_sp = [int(row.strip('%')) for row in cross_table_sp.index]

            precision_sorted_se = sorted(precision_se)
            conf_levels_sorted_se = sorted(conf_levels_se)
            precision_sorted_sp = sorted(precision_sp)
            conf_levels_sorted_sp = sorted(conf_levels_sp)


            power_labels_se = [f"{p}%" for p in precision_sorted_se]
            conf_labels_se = [f"{cl}%" for cl in conf_levels_sorted_se]
            power_labels_sp = [f"{p}%" for p in precision_sorted_sp]
            conf_labels_sp = [f"{cl}%" for cl in conf_levels_sorted_sp]

            col1, col2 = st.columns(2)

            # === Plot 1: Sample Size vs Precision at fixed Confidence Levels ===
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                conf_levels_to_plot = [80, 95, 97, 99]
                for cl in conf_levels_to_plot:
                    cl_label = f"{cl}%"
                    if cl_label in cross_table_se.index:
                        sample_sizes = cross_table_se.loc[cl_label, power_labels_se].astype(float).tolist()
                        ax1.plot(precision_sorted_se, sample_sizes, marker='o', linestyle='-', label=f'CL {cl_label}')
                ax1.set_title("Sample Size vs Precision for Sensitivity")
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
                    if pr_label in cross_table_se.columns:
                        sample_sizes = cross_table_se[pr_label].astype(float).tolist()
                        ax2.plot(conf_levels_sorted_se, sample_sizes, marker='s', linestyle='--', label=f'Precision {pr_label}')
                ax2.set_title("Sample Size vs Confidence Level for Sensitivity")
                ax2.set_xlabel("Confidence Level (%)")
                ax2.set_ylabel("Sample Size")
                ax2.grid(True)
                ax2.legend(title="Precision")
                st.pyplot(fig2)


            coll1, coll2 = st.columns(2)

            # === Plot 1: Sample Size vs Precision at fixed Confidence Levels ===
            with coll1:
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                conf_levels_to_plot = [80, 95, 97, 99]
                for cl in conf_levels_to_plot:
                    cl_label = f"{cl}%"
                    if cl_label in cross_table_sp.index:
                        sample_sizes = cross_table_sp.loc[cl_label, power_labels_sp].astype(float).tolist()
                        ax1.plot(precision_sorted_sp, sample_sizes, marker='o', linestyle='-', label=f'CL {cl_label}')
                ax1.set_title("Sample Size vs Precision for Specificity")
                ax1.set_xlabel("Precision (%)")
                ax1.set_ylabel("Sample Size")
                ax1.grid(True)
                ax1.legend(title="Confidence Level")
                st.pyplot(fig1)

            # === Plot 2: Sample Size vs Confidence Level at fixed Precision Levels ===
            with coll2:
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                precision_levels_to_plot = [2, 3, 5, 10]
                for pr in precision_levels_to_plot:
                    pr_label = f"{pr}%"
                    if pr_label in cross_table_sp.columns:
                        sample_sizes = cross_table_sp[pr_label].astype(float).tolist()
                        ax2.plot(conf_levels_sorted_sp, sample_sizes, marker='s', linestyle='--', label=f'Precision {pr_label}')
                ax2.set_title("Sample Size vs Confidence Level for Specificity")
                ax2.set_xlabel("Confidence Level (%)")
                ax2.set_ylabel("Sample Size")
                ax2.grid(True)
                ax2.legend(title="Precision")
                st.pyplot(fig2)


            st.markdown("---")
            with st.expander("💡Show the Interpretation of the plots"):
                st.markdown("### Plot 1 & 3: Sample Size vs Precision")
                st.markdown("- As **precision becomes tighter (i.e., smaller %) the required sample size increases** exponentially.")
                st.markdown("- Higher confidence levels (e.g., 99%) require larger sample sizes than lower ones (e.g., 80%) for the same precision.")
                st.markdown("### Plot 2 & 4: Sample Size vs Confidence Level")
                st.markdown("- As **confidence level increases**, so does the **required sample size** to ensure the estimate remains within the desired precision.")
                st.markdown("- At lower confidence (e.g., 70–80%), sample size requirements are modest, but they grow rapidly beyond 95%, especially at tighter precision levels.")

    st.markdown("---")  # Adds a horizontal line for separation
    with st.expander("Show the formula and the references"):
        st.subheader("📌 Formula for Sample Size Calculation")

        st.markdown("### **Sensitivity Sample Size Formula**")
        st.latex(r"""
        n_{Se} = \left( \frac{Z_{1-\alpha/2}}{d \times p} \right)^2 \times Se (1 - Se) \times \frac{DE}{1 - \text{Dropout\%}}
        """)

        st.markdown("### **Specificity Sample Size Formula**")
        st.latex(r"""
        n_{Sp} = \left( \frac{Z_{1-\alpha/2}}{d \times (1-p)} \right)^2 \times Sp (1 - Sp) \times \frac{DE}{1 - \text{Dropout\%}}
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
        - **Dropout%**: Anticipated percentage of dropout in the study.
        """)

        st.subheader("References")

        st.markdown("""
        1. **Buderer, N. M. F. (1996).** Statistical methodology: I. Incorporating the prevalence of disease into the sample size calculation for sensitivity and specificity. Acadademic Emergency Medicine, 3(9), 895-900. Available at: [https://pubmed.ncbi.nlm.nih.gov/8870764/](https://pubmed.ncbi.nlm.nih.gov/8870764/)
        """)

    st.markdown("---")
    st.subheader("Citation")
    from datetime import datetime
    # Get current date and time
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    # Citation with access timestamp
    st.markdown(f"""
    *Majumder, R. (2025). StudySizer: A sample size calculator (Version 0.1.0). Available online: [https://studysizer.streamlit.app/](https://studysizer.streamlit.app/). Accessed on {now}. [https://doi.org/10.5281/zenodo.16375937](https://doi.org/10.5281/zenodo.16375937).*
    """)


    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")