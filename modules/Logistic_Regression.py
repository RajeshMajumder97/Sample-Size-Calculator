import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

def main():
    #    st.set_page_config(page_title="StydySizer | Logistic Regression", page_icon="🧮")
    #
    st.title("Sample Size Calculation for Multiple Logistic Regression")
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

    # Formula function
    def nSampleLogisticRegression(P=0.1, OR=1.5, R2=0.2, alpha=0.05, power=0.8, k=1, designEf=1, dropOut=0.0):
        Z_alpha = norm.ppf(1 - alpha / 2)
        Z_beta = norm.ppf(power)

        log_OR = np.log(OR)
        q = 1 - P
        numerator = (Z_alpha + Z_beta) ** 2 * (1 + (1 - R2) * (k - 1))
        denominator = P * q * (log_OR ** 2) * (1 - R2)
        n = numerator / denominator
        n_adjusted = n / (1 - dropOut)
        return int(np.ceil(n_adjusted * designEf))

    # Initialize session history
    if "logr_history" not in st.session_state:
        st.session_state.logr_history = []

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Input Parameters")

    # Sidebar inputs
    P = st.sidebar.number_input("Overall Proportion of Disease", value=10.0, min_value=1.0, max_value=99.99)
    OR = st.sidebar.number_input("Anticipated Odds Ratio (OR)", value=1.5, min_value=0.01)
    R2 = st.sidebar.number_input("R-squared with other predictors (R²)", value=0.2, min_value=0.0, max_value=0.99,help="Enter a decimal value (e.g., 0.05)")
    power = st.sidebar.number_input("Power (%)", value=80.0, min_value=50.0, max_value=99.9,help="Enter a percentage value (e.g., 80%)")
    k = st.sidebar.number_input("Number of Predictors (k)", value=1, min_value=1)
    drp = st.sidebar.number_input("Drop-Out (%)", value=0.0, min_value=0.0, max_value=50.0,help="Enter a percentage value (e.g., 1%)")    

    method = st.sidebar.radio("Choose Method for Design Effect:", options=["Given", "Calculate"])

    if method == "Given":
        designEffect = st.sidebar.number_input("Design Effect (Given)", value=1.0, min_value=1.0,help="Enter a decimal value (e.g., 1.5)")
        m = None
        ICC = None
    else:
        m = st.sidebar.number_input("Number of Clusters (m)", min_value=2,value=4,help="Enter an integer value (e.g., 4)")
        ICC = st.sidebar.number_input("Intra-class Correlation (ICC) for clustering", min_value=0.0,max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
        designEffect = 1 + (m - 1) * ICC
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster Size (m)", value=m)
        col2.metric("ICC", value=ICC)
        col3.metric("Design Effect", value=round(designEffect, 2))

    go = st.button("Calculate Sample Size")

    def make_logr_history_label(P, OR, R2, power, k, drp, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"P={P}, OR={OR}, R²={R2}, Power={power}%, k={k}, DropOut={drp}%, DE={round(designEffect,2)}"
        else:
            return f"P={P}, OR={OR}, R²={R2}, Power={power}%, k={k}, DropOut={drp}%, DE(Calc)={round(designEffect,2)}, m={m}, ICC={ICC}"

    selected_history = None
    if st.session_state.logr_history:
        st.subheader("📜 Select from Past Inputs")
        hist_labels = [make_logr_history_label(**entry) for entry in st.session_state.logr_history]
        selected = st.selectbox("Choose a past input set:", hist_labels, key="logr_history_selector")
        if selected:
            selected_history = next((item for item in st.session_state.logr_history if make_logr_history_label(**item) == selected), None)
            recalc = st.button("🔁 Recalculate")
        else:
            recalc = False
    else:
        recalc = False

    if go or recalc:
        tabs = st.tabs(["Tabulate", "Power V/s Confidence Table" ,"Visualisation"])
        with tabs[0]:
            if recalc and selected_history:
                P = selected_history["P"]
                OR = selected_history["OR"]
                R2 = selected_history["R2"]
                power = selected_history["power"]
                #alpha = selected_history["alpha"]
                drp = selected_history["drp"]
                k = selected_history["k"]
                designEffect = selected_history["designEffect"]
            else:
                st.session_state.logr_history.append({
                    "P": P, "OR": OR, "R2": R2, "power": power,
                    "drp": drp, "k": k,
                    "designEffect": designEffect, "m": m, "ICC": ICC, "method": method
                })

            conf_levels = [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
            results = [nSampleLogisticRegression(P/100  , OR, R2, 1 - cl, power / 100, k, designEffect, drp / 100) for cl in conf_levels]

            df = pd.DataFrame({"Confidence Level (%)": [(c * 100) for c in conf_levels], "Sample Size": results})
            n95 = nSampleLogisticRegression(P/100, OR, R2, 0.95, power / 100, k, designEffect, drp / 100)

            st.write("The required sample size is:")
            st.markdown(f"""
            <div style='display: flex; justify-content: center;'>
                <div style='font-size: 36px; font-weight: bold; background-color: #48D1CC; padding: 10px; border-radius: 10px;'>
                    {n95}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"""
            To detect an Odds Ratio of {OR} for a predictor in a multiple logistic regression model with overall disease proportion {P}, R²={R2}, {power}% power, and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level. Design effect: {round(designEffect,2)}, Drop-out: {drp}%.
            """, unsafe_allow_html=True)

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
                    ss =   nSampleLogisticRegression(P/100  ,  OR, R2, (1 - conf), power_val, k, designEffect, drp / 100)
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


    # Math and parameters
    st.markdown("---")
    with st.expander("Show the formula and the references"):
        st.subheader("📌 Formula for Sample Size Calculation") 
        st.latex(r"""
        n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (1 + (1 - R^2)(k - 1))}{P(1 - P)(\ln(OR))^2(1 - R^2)} \times \frac{DE}{1 - \text{Dropout\%}}
        """)

        st.markdown("### Design Effect (if clusters are used):")
        st.latex(r"""
        DE = 1 + (m - 1) \cdot ICC
        """)

        st.subheader("📌 Description of Parameters")

        st.markdown("""
        - **\( P \)**: Overall disease prevalence or proportion of success.
        - **\( OR \)**: Anticipated Odds Ratio.
        - **\( R^2 \)**: Multiple correlation coefficient of exposure with other covariates.
        - **\( \alpha \)**: Significance level (typically 0.05).
        - **\( \beta \)**: Type II error = 1 - Power.
        - **\( k \)**: Number of predictors in the model.
        - **\( DE \)**: Design effect (for cluster sampling).
        - **Dropout%**: Anticipated percentage of dropout in the study.
        """)

        st.subheader("📌 References")
        st.markdown("""
        1. **Hsieh FY, Bloch DA, Larsen MD. (1998)** A simple method of sample size calculation for linear and logistic regression. Statistics in Medicine.
        2. **Dupont WD, Plummer WD. (1998)** PS Power and Sample Size Calculations. Controlled Clinical Trials.
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
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app](https://rajeshmajumderblog.netlify.app)")
