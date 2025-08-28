import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm


def main():

    #    st.set_page_config(page_title="StydySizer | Survival Analysis | Log-rank Test Sample Size (Two Groups)", page_icon="🧮")
    #
    st.title("Sample Size Calculation for Survival Analysis | Log-rank Test (Two Sample)")
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
    # Sample size calculation function for log-rank test
    def nSampleSurvival(HR=0.7, Pw=0.8, Conf=0.95, p=0.5, eventRate=0.6, designEf=1.0, dropOut=0.0):
        z_alpha = norm.ppf(1 - (1 - Conf) / 2)
        z_beta = norm.ppf(Pw)
        n_events = ((z_alpha + z_beta) ** 2) / (np.log(HR) ** 2 * p * (1 - p))
        n_total = n_events / eventRate
        return int(np.ceil((n_total / (1 - dropOut)) * designEf))

    if "survival_history" not in st.session_state:
        st.session_state.survival_history = []


    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")

    # Sidebar inputs
    HR = st.sidebar.number_input("Hazard Ratio (HR)", value=0.7, min_value=0.001,format="%.6g",help="Enter a decimal value (e.g., 0.05)")
    power = st.sidebar.number_input("Power (%)", value=80.0, min_value=0.0, max_value=99.99,format="%.6g",help="Enter a percentage value (e.g., 80%)")
    #conf = st.sidebar.number_input("Confidence Level (%)", value=95.0, min_value=50.0, max_value=99.999)
    p = st.sidebar.number_input("Allocation Ratio (Group 1) [Exposed Group]", value=0.5, min_value=0.01, max_value=0.99,format="%.6g",help="Enter a decimal value (e.g., 0.5)")
    eventRate = st.sidebar.number_input("Expected Event Rate", value=0.6, min_value=0.01, max_value=1.0,format="%.6g",help="Enter a decimal value (e.g., 0.6)")
    drp = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=50.0,format="%.6g",help="Enter a percentage value (e.g., 1%)")

    method = st.sidebar.radio("Choose Method for Design Effect:", options=["Given", "Calculate"])

    if method == "Given":
        designEffect = st.sidebar.number_input("Design Effect (Given)", value=1.0, min_value=1.0,format="%.6g",help= "Enter an decimal value (e.g., 1.5)")
        m = None
        ICC = None
    else:
        m = st.sidebar.number_input("Number of Clusters (m)", min_value=2,value=4,format="%.6g",help="Enter an integer value (e.g., 4)")
        ICC = st.sidebar.number_input("Intra-class Correlation (ICC) for clustering", min_value=0.0,max_value=1.0,value=0.05,format="%.6g",help="Enter a decimal value (e.g., 0.05)")
        designEffect = 1 + (m - 1) * ICC
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster Size (m)", value=m)
        col2.metric("ICC", value=ICC)
        col3.metric("Design Effect", value=round(designEffect, 2))

    # Calculate button
    go = st.button("Calculate Sample Size")

    # History label
    def make_survival_label(HR, power, p, eventRate, drp, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"HR={HR}, Power={power}%, EventRate={eventRate}, DE={round(designEffect, 2)}"
        else:
            return f"HR={HR}, Power={power}%, EventRate={eventRate}, DE={round(designEffect, 2)}, m={m}, ICC={ICC}"

    # Select from history
    selected_history = None
    if st.session_state.survival_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate")
        survival_labels = [make_survival_label(**item) for item in st.session_state.survival_history]
        selected = st.selectbox("Choose a past input set:", survival_labels, key="survival_history_selector")
        if selected:
            selected_history = next(item for item in st.session_state.survival_history if make_survival_label(**item) == selected)
            recalc = st.button("🔁 Recalculate from Selected History")
        else:
            recalc = False
    else:
        recalc = False

    if go or recalc:
        tabs = st.tabs(["Tabulate", "Power V/s Confidence Table" ,"Visualisation"])
        with tabs[0]:
            if recalc and selected_history:
                HR = selected_history["HR"]
                power = selected_history["power"]
                #conf = selected_history["conf"]
                p = selected_history["p"]
                eventRate = selected_history["eventRate"]
                drp = selected_history["drp"]
                designEffect = selected_history["designEffect"]
            else:
                st.session_state.survival_history.append({
                    "HR": HR,
                    "power": power,
                    #"conf": conf,
                    "p": p,
                    "eventRate": eventRate,
                    "drp": drp,
                    "designEffect": designEffect,
                    "m": m,
                    "ICC": ICC,
                    "method": method
                })

            conf_levels = [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
            results = []
            for conf in conf_levels:
                n = nSampleSurvival(HR=HR, Conf= conf, Pw=power / 100,p=p, eventRate=eventRate, designEf=designEffect, dropOut=drp / 100)
                results.append(n)

            df = pd.DataFrame({"Confidence Level (%)": [(c * 100) for c in conf_levels], "Sample Size": results})

            sample_size = nSampleSurvival(HR=HR, Pw=power / 100, Conf=(0.95), p=p, eventRate=eventRate, designEf=designEffect, dropOut=drp / 100)

            st.write("The required total sample size is:")
            st.markdown(f"""
            <div style='display: flex; justify-content: center;'>
                <div style='font-size: 36px; font-weight: bold; background-color: #48D1CC; padding: 10px; border-radius: 10px;'>
                    {sample_size}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"""To detect a hazard ratio of {HR} with {power}% power and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, assuming event rate of {eventRate}, allocation ratio {p}, design effect of {round(designEffect, 2)} and drop-out of {drp}%.""", unsafe_allow_html=True)

            st.subheader("Sample Sizes at Other Confidence Levels")
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
                    ss =  nSampleSurvival(HR=HR, Conf= conf, Pw=power_val,p=p, eventRate=eventRate, designEf=designEffect, dropOut=drp / 100)
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

        

    st.markdown("---")
    with st.expander("Show the formula and the references"):
        st.subheader("📌 Formula for Sample Size Calculation")
        st.latex(r"""
        n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2}{[\log(HR)]^2 \cdot p(1-p)} \times \frac{1}{\text{event rate}} \times \frac{DE}{1 - \text{Dropout\%}}
        """)

        st.markdown("### Design Effect (if clusters are used):")
        st.latex(r"""
        DE = 1 + (m - 1) \times ICC
        """)

        st.subheader("📌 Description of Parameters")
        st.markdown("""
        - **\( HR \)**: Hazard Ratio to detect
        - **\( p \)**: Allocation proportion (Group 1) [Exposed Group]
        - **\( Z_{1-alpha/2} \)**: Z-score for confidence level
        - **\( Z_{1-beta} \)**: Z-score for power
        - **\( event\ rate \)**: Proportion of individuals expected to experience the event
        - **\( DE \)**: Design effect
        - **\( ICC \)**: Intra-cluster correlation.
        - **Dropout%**: Anticipated percentage of dropout in the study.
        """)

        st.markdown("---")
        st.markdown("""
            <div style="
                background-color: #48D1CC;
                padding: 10px;
                border-left: 5px solid orange;
                border-radius: 5px;
                font-size: 18px;">
                <b>Note:</b> This tool calculates the expected number of events by default. Multiply by 𝑑(event rate[0-1)) if you are calculating the expected number of events from a given sample size (default option) and Divide by 𝑑 if you are calculating the required sample size from a given number of events. Suppose your calculation shows that you need 100 events to detect a hazard ratio difference with the desired statistical power. If your estimated event rate is 40% (i.e., only 40% of participants are expected to experience the event), then you will need to enroll 100 divided by 0.4, which equals 250 participants in total. Conversely, if you are planning to enroll 250 participants in the study and you expect an event rate of 40%, then the expected number of events will be 250 multiplied by 0.4, which equals 100 events.
            </div>
            """, unsafe_allow_html=True)


        st.markdown("---")
        st.subheader("References")
        st.markdown("""
        1. Freedman, L. S. (1982). Tables of the number of patients required in clinical trials using the log-rank test. *Stat Med.*
        2. Chow, S. C., Shao, J., & Wang, H. (2008). Sample Size Calculations in Clinical Research.
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
