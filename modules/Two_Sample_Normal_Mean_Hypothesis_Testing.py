import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

def main():
    #    st.set_page_config(page_title="StydySizer | Two Sample Normal Mean Hypothesis Testing",
    #                    page_icon="🧮")
    #
    # Streamlit App
    st.title("Sample Size Calculation for Two sample Mean Test | H0: Mu1=Mu2")

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
    def nSampleMean(sigma=0.01,Pw=0.8,delta=0.05,Conf=0.95,designEf=1,dropOut=0):
        n= (2*(norm.ppf(1-(1-Conf)/2)+norm.ppf(Pw))**2)*(sigma/delta)**2                               
        return(abs(round((n/(1-dropOut))*designEf)))

    # Initialize history store
    if "mean_test_history" not in st.session_state:
        st.session_state.mean_test_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")

    sigma = st.sidebar.number_input("Standard Deviation (SD)",value=15.0,min_value=0.01,help= "Enter a value >0")
    delta = st.sidebar.number_input("Expected difference", value=10.0,min_value=0.0,help= "Enter a value >0")
    power= st.sidebar.number_input("Power (%)", value=80.0,min_value=0.0,max_value=99.99,help="Enter a percentage value (e.g., 80%)")
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
        tabs = st.tabs(["Tabulate", "Power V/s Confidelce Table" ,"Visualisation"])

        with tabs[0]:
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

            confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
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
                    ss = nSampleMean(sigma=sigma,delta=delta,Pw=power_val,Conf=conf,designEf=designEffect,dropOut=(drpt/100))  
                    cross_table.iloc[i, j] = ss

            # Label table
            cross_table.index.name = "Confidence Level (%)"
            cross_table.columns.name = "Power (%)"

            st.dataframe(cross_table)
            st.write("**Rows are Confidence Levels; Columns are Powers**")
            #st.session_state["cross_table"] = cross_table
        with tabs[2]:
            ##
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