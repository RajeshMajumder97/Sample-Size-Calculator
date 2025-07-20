import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from scipy.special import erf

def main():
    #    st.set_page_config(page_title="StydySizer | Intraclass Correlation",
    #                    page_icon="🧮")
    #

    # Streamlit App
    st.title("Sample Size Calculation for Intraclass Correlation Hypothesis Testing")
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

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")
    Obj = st.sidebar.number_input("Observation/Subject (n)",value=5,min_value=0,help= "Enter an integer value (e.g., 5)")
    st.sidebar.text("Number of repeted observatiuons\n by different judges\n per subject,replicates")
    power= st.sidebar.number_input("Power (%)",value=80.0,min_value=50.0,max_value=99.9,help= "Enter a percentage value (e.g., 80%)")
    minAR= st.sidebar.number_input("Minimum acceptable reliability (ρ₀) (%)",value=60.0,min_value=0.0,max_value=100.0,help= "Enter a percentage value (e.g., 60%)")
    st.sidebar.text("The lowest limit of reliability\n you would accept")
    ERR= st.sidebar.number_input("Expected reliability (ρ₁) (%)",value=80.0,min_value=0.0,max_value=100.0,help= "Enter a percentage value (e.g., 60%)")
    st.sidebar.text("The level of reliability\n you can expect from the study")
    drpt= st.sidebar.number_input("Drop-Out (%)",value=0.0,min_value=0.0,max_value=50.0,help= "Enter a percentage value (e.g., 1%)")

    x= st.sidebar.radio("Choose Method for Design Effect:",options=['Given','Calculate'])

    if(x== "Given"):
        designEffect= st.sidebar.number_input("Design Effect (Given)", value=1.0,min_value=1.0,help= "Enter a decimal value (e.g., 1.5)")
        m=None
        ICC=None
    else:
        m= st.sidebar.number_input("Number of Cluster (m)",min_value=2,value=4,help="Enter an integer value (e.g., 4)")
        ICC= st.sidebar.number_input("Intra-class Correlation (ICC) for clustering",min_value=0.0,max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
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
        tabs = st.tabs(["Tabulate", "Power V/s Confidelce Table" ,"Visualisation"])
        with tabs[0]:
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

            confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
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
                    ss = nSampleICC(n=Obj,rho0=(minAR/100),rho1=(ERR/100),Conf=conf,Pw=power_val,designEf=designEffect,dropOut=(drpt/100)) 
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

        st.markdown("### **Sample Size Formula for Intraclass Correlation (ICC) Estimation**")

        st.latex(r"""
        N = \frac{1 + 2(Z_{\alpha} + Z_{\beta})^2 \cdot n}{\left(\ln\left(\frac{1 + \frac{n \rho_0}{1 - \rho_0}}{1 + \frac{n \rho_1}{1 - \rho_1}}\right)\right)^2 (n - 1)} \times \frac{DE}{1 - \text{Dropout\%}}
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
        - **Dropout%**: Anticipated percentage of dropout in the study.
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