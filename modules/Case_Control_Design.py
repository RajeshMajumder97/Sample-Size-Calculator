import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf
import math

def main():


    # Streamlit App
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

    p2 = st.sidebar.number_input("Proportion of Exposed among controls (%)",value=40.0,min_value=0.0,max_value=99.9,format="%.6g",help="Enter a percentage value (e.g., 40%)")
    R = st.sidebar.number_input("Anticipated odds ratio (OR)", value=1.5,min_value=0.00001,format="%.6g",help= "plausible range > or < 1.")

    if R == 1.0:
        st.sidebar.warning("OR should not be 1.0 under H₁. Change it to reflect a real difference.")
        st.stop()

    power= st.sidebar.number_input("Power (%)", value=80.00,min_value=50.0,max_value=99.9,format="%.6g",help="Enter a percentage value (e.g., 80%)")
    drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=50.0,format="%.6g",help="Enter a percentage value (e.g., 1%)")


    x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

    if(x== "Given"):
        designEffect= st.sidebar.number_input("Design Effect (Given)", value=1.0,min_value=1.0,format="%.6g",help= "Enter a decimal value (e.g., 1.5)")
        m=None
        ICC=None
    else:
        m= st.sidebar.number_input("Number of Clusters (m) ",min_value=2,value=4,format="%.6g",help="Enter an integer value (e.g., 4)")
        ICC= st.sidebar.number_input("Intra-class Correlation (ICC) for clustering",min_value=0.0,format="%.6g",max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
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
        tabs = st.tabs(["Tabulate", "Power V/s Confidence Table" ,"Visualisation"])
        with tabs[0]:
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

            confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
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

        with tabs[1]:
            # D efine power and confidence levels
            powers = [0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]
            conf_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

            st.subheader("📈 Sample Size Cross Table for Different Powers and Confidence Levels")

            power_labels = [f"{int(p * 100)}%" for p in powers]
            conf_labels = [f"{int(c * 100)}%" for c in conf_levels]
            cross_table = pd.DataFrame(index=conf_labels, columns=power_labels)
            # Fill the cross table
            for i, conf in enumerate(conf_levels):
                for j, power_val in enumerate(powers):
                    ss = nSampleOR(p2=(p2/100),OR=R,Pw=power_val,Conf=conf,designEf=designEffect,dropOut=(drpt/100)) 
                    cross_table.iloc[i, j] = ss
            # Label table
            cross_table.index.name = "Confidence Level (%)"
            cross_table.columns.name = "Power (%)"

            st.dataframe(cross_table)
            st.write("**Rows are Confidence Levels; Columns are Powers**")
            #st.session_state["cross_table"] = cross_table
        with tabs[2]:
            import matplotlib.pyplot as plt

            powers = [int(col.strip('%')) for col in cross_table.columns]
            conf_levels = [int(row.strip('%')) for row in cross_table.index]

            # Sort both for consistent plotting
            powers_sorted = sorted(powers)
            conf_levels_sorted = sorted(conf_levels)

            # Convert back to string labels
            power_labels = [f"{p}%" for p in powers_sorted]
            conf_labels = [f"{cl}%" for cl in conf_levels_sorted]

            # Plotting
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Power curves at selected Confidence Levels (primary y-axis)
            conf_levels_to_plot = [90, 95, 97, 99]
            for cl in conf_levels_to_plot:
                cl_label = f"{cl}%"
                if cl_label in cross_table.index:
                    sample_sizes = cross_table.loc[cl_label, power_labels].astype(float).tolist()
                    ax1.plot(sample_sizes, powers_sorted, marker='o', linestyle='-', label=f'Power at {cl_label} CL')

            ax1.set_xlabel("Sample Size")
            ax1.set_ylabel("Power (%)", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim([60, 100])
            ax1.grid(True)

            # Alpha curves at selected Power Levels (secondary y-axis)
            power_levels_to_plot = [80, 85, 90, 95]
            ax2 = ax1.twinx()
            for pwr in power_levels_to_plot:
                pwr_label = f"{pwr}%"
                if pwr_label in cross_table.columns:
                    sample_sizes = cross_table[pwr_label].astype(float).tolist()
                    alpha_vals = [100 - int(idx.strip('%')) for idx in cross_table.index]
                    ax2.plot(sample_sizes, alpha_vals, marker='s', linestyle='--', label=f'Alpha at {pwr_label} Power')

            ax2.set_ylabel("Alpha Level (%)", color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim([0, 30])

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            #ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1.05, 0.5), ncol=2)

            # Title and layout
            plt.title("Sample Size vs Power and Alpha Level (Multiple Lines)")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)
            #plt.tight_layout(rect=[0, 0.1, 1, 1])
            # Show in Streamlit
            st.pyplot(fig)
            st.markdown("---")
            with st.expander("💡Show the Interpretation of the plot"):
                st.markdown("- This plot demonstrates **how sample size influences both statistical power and the risk of Type I error (alpha)**—two critical factors in designing reliable health research.")
                st.markdown("- The **Left Y-Axis (Blue)**, solid lines represent the probability of correctly detecting a true effect (power), which increases with larger sample sizes, improving the study's ability to identify meaningful (clinical) differences.")
                st.markdown("- On the other hand, the **Right Y-Axis (Orange/Yellow)**, dashed lines indicate the likelihood of a false positive result (alpha), which typically decreases with larger samples, reflecting a more conservative test. Conversely, increasing alpha reduces the required sample size to achieve a given power, but increases the risk of Type I error. For an example, if you want 80% power, increasing alpha (e.g., from 0.01 to 0.05) means you need fewer subjects.")
                st.markdown("- **Points where the power and alpha curves intersect** represent sample sizes where the chance of detecting a real effect (power) equals the chance of making a false claim (alpha)—an undesirable scenario. In health research, we strive for power to be much higher than alpha to ensure that findings are both valid and clinically trustworthy, in line with the principles of the most powerful statistical tests. ")

    st.markdown("---")  # Adds a horizontal line for separation
    with st.expander("Show the formula and the references"):
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