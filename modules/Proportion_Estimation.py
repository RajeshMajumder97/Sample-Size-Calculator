import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf


def main():
    #    st.set_page_config(page_title="StydySizer | Proportion Estimation",
    #                    page_icon="🧮")
    #
    # Streamlit App
    st.title("Sample Size Calculation for Proportion: Proportion Estimation")
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
    def nSampleProp(p=0.5,d=0.05,Conf=0.95,designEf=1,dropOut=0):
        n= ((norm.ppf(1-(1-Conf)/2)/d)**2)*(p*(1-p))
        return(abs(round((n/(1-dropOut))*designEf)))

    # Initialize history store
    if "pp_history" not in st.session_state:
        st.session_state.pp_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")

    p = st.sidebar.number_input("Expected Proportion (%)",value=50.0,min_value=0.0,max_value=99.99,help="Enter a percentage value (e.g., 50%)")
    d = st.sidebar.number_input("Precision or Margin of Error (%)",min_value=0.0, value=10.0,max_value=50.0,help="Enter a percentage value (e.g., 10%).")
    ads= st.sidebar.radio("Choose Precision or Margin of Error Option",options=['Absolute Precision','Relative to the Proportion'],help="This represents how precisely you want to estimate the true proportion in the population. A smaller margin of error leads to a larger required sample size and a narrower confidence interval. For instance, suppose a clinical survey finds that 30% of patients report improvement after taking a new medication. If we use a 5% 'absolute precision', we can say with confidence that the true proportion of patients who benefit lies between 25% (30−5) and 35% (30+5). However, if we use a 5% 'relative precision', the confidence range becomes 28.5% (30−5% of 30) to 31.5% (30+5% of 30). The choice between absolute and relative precision affects how narrowly we can define the likely range of the true effect in the broader patient population.")

    if(ads=='Absolute Precision'):
        d1=d
    else:
        d1= ((d/100)*(p/100))*100

    drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=50.0,help="Enter a percentage value (e.g., 1%)")

    x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

    if(x== "Given"):
        designEffect= st.sidebar.number_input("Design Effect (Given)", value=1.0,min_value=1.0,help= "Enter an decimal value (e.g., 1.5)")
        m=None
        ICC=None
    else:
        m= st.sidebar.number_input("Number of clusters (m)",min_value=2,value=4, help="Enter an integer value (e.g., 4)")
        ICC= st.sidebar.number_input("Intra-class Correlation (ICC) for clustering",min_value=0.0,max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
        designEffect= 1+(m-1)*ICC
        col1,col2,col3=st.columns(3)
        col1.metric("Cluster Size (m)",value=m)
        col2.metric("Intra Class Correlation (ICC)",value=ICC)
        col3.metric("Design Effect",value= round(designEffect,2))

    # Calculate button
    go = st.button("Calculate Sample Size")

    # Helper to generate label for dropdown
    def make_pp_history_label(p, d1, drpt, designEffect, m=None, ICC=None, method="Given",absolute='Absolute Precision',d=None):
        if method == "Given":
            if absolute=='Absolute Precision':
                return f"Preval={p}%, Precision method={absolute}, Precision(abs)={round(d1,2)}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
            else:
                return f"Preval={p}%,  Precision method={absolute},Precision(relt({d}%))={round(d1,2)}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            if absolute=='Absolute Precision':
                return (f"Preval={p}%,  Precision method={absolute},Precision(abs)={d1}%, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")
            else:
                return (f"Preval={p}%,  Precision method={absolute},Precision(relt({d}%))={d1}%, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")            

    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.pp_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        pp_options = [make_pp_history_label(**entry) for entry in st.session_state.pp_history]
        selected_label = st.selectbox("Choose a past input set:", pp_options, key="pp_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.pp_history
                                    if make_pp_history_label(**item) == selected_label), None)
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
                d1= selected_history["d1"]
                drpt= selected_history["drpt"]
                ads= selected_history["absolute"]
                designEffect = selected_history["designEffect"]
            else:
                # Add current input to history
                new_entry = {
                    "p":p,
                    "d1":d1,
                    "drpt":drpt,
                    "designEffect":designEffect,
                    "m":m,
                    "ICC":ICC,
                    "method":x,
                    "absolute": ads,
                    "d":d
                }
                st.session_state.pp_history.append(new_entry)

            confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
            out=[]

            for conf in confidenceIntervals:
                sample_size= nSampleProp(p=(p/100),d=(d1/100),Conf=conf,designEf=designEffect,dropOut=(drpt/100))
                out.append(sample_size)

            df= pd.DataFrame({
                "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
                "Sample Size": out
            })
            dds= nSampleProp(p=(p/100),d=(d1/100),Conf=0.95,designEf=designEffect,dropOut=(drpt/100))
            if(ads=='Absolute Precision'):
                st.write(f"Asuming that **{(p)}%** of the individuals in the population exhibit the characteristic of interest, the study would need a sample size of:")
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
                st.write(f"""participants to estimate the expected proportion with an absolute precision of **{(d1)}%** and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence interval, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)
            else:
                st.write(f"Asuming that **{(p)}%** of the individuals in the population exhibit the characteristic of interest, the study would need a sample size of:")
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
                st.write(f"""participants to estimate the expected proportion with an absolute precision of **({(p)}% * {(d)}%) = {round(d1,1)}** and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence interval, considering a design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out from the sample.""",unsafe_allow_html=True)

            st.subheader("List of Sample Sizes at other Confidence Levels")
            st.dataframe(df)
        with tabs[1]:
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
                    if(ads=='Absolute Precision'):
                        d1=d_val
                    else:
                        d1= (d_val*(p/100))
                    ss = nSampleProp(p=(p/100),d=d1,Conf=conf,designEf=designEffect,dropOut=(drpt/100)) 
                    cross_table.iloc[i, j] = ss
            # Label table
            cross_table.index.name = "Confidence levels (%)"
            cross_table.columns.name = "Precision(%)"
            st.dataframe(cross_table)
            st.write("**Rows are Confidence levels; Columns are Precision**")
            #st.session_state["cross_table"] = cross_table
        with tabs[2]:
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

        st.markdown("### **Proportion-Based Sample Size Formula**")
        st.latex(r"""
        n = \left( \frac{Z_{1-\alpha/2}}{d} \right)^2 \times p (1 - p) \times \frac{DE}{1 - \text{Dropout\%}}
        """)

        st.markdown("### **Design Effect Calculation (if clusters are used):**")
        st.latex(r"""
        DE = 1 + (m - 1) \times ICC
        """)

        st.subheader("📌 Description of Parameters")

        st.markdown("""
        - **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
        - **\( d \)**: Precision (margin of error).
        - **\( p \)**: Expected proportion.
        - **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
        - **\( m \)**: Number of cluster.
        - **\( ICC \) (Intra-cluster correlation coefficient)**: Measures similarity within clusters.
        - **Dropout%**: Anticipated percentage of dropout in the study.
        """)

        #st.markdown("""
        #    <div style="
        #        background-color: #f9f871;
        #        padding: 10px;
        #        border-left: 5px solid orange;
        #       border-radius: 5px;
        #        font-size: 18px;">
        #        <b>Note:</b> The design effect option is only applicable when doing cluster random sampling, other wise the default is 1 and it is recommended to be done in consultation with a statistician.   
        #    </div>
        #    """, unsafe_allow_html=True)


        st.subheader("📌 References")

        st.markdown("""
            1. Lwanga, S. K., & Lemeshow, S. (1991). Sample size determination in health studies: A practical manual. WHO.
            2. Daniel, W. W. (1999). Biostatistics: A Foundation for Analysis in the Health Sciences. 7th ed. Wiley.
        """)



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