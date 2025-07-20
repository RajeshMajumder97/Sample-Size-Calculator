import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

def main():
    # Streamlit App Setup
#    st.set_page_config(page_title="StydySizer | Two Sample Proportion Hypothesis Testing", page_icon="🧮")
#
    # Title
    st.title("Sample Size Calculation for Proportion Test | H0: P1=P2")
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
    # Function to calculate sample size
    def nSampleProp(p1=0.5, p2=0.5, delta=0.0, Pw=0.8, Conf=0.95, designEf=1, dropOut=0):
        n = (((norm.ppf(1 - (1 - Conf) / 2) + norm.ppf(Pw)) ** 2) * (p1 * (1 - p1) + p2 * (1 - p2))) / (delta ** 2)
        return abs(round((n / (1 - dropOut)) * designEf))

    # Initialize history store
    if "prop_test_history" not in st.session_state:
        st.session_state.prop_test_history = []
    if "go_clicked" not in st.session_state:
        st.session_state.go_clicked = False
    if "submit_clicked" not in st.session_state:
        st.session_state.submit_clicked = False


    def handle_go():
        st.session_state.go_clicked = True
        st.session_state.submit_clicked = False

    def handle_submit():
        st.session_state.submit_clicked = True
        st.session_state.go_clicked = False

        

    st.sidebar.markdown("---")
    with st.sidebar.form("input_form"):
        st.markdown("### 🔧 Input Parameters")

        p1 = st.number_input("Proportion in 1st (Reference) Group (%)", value=50.0, min_value=0.0, max_value=99.99, help="Enter a percentage value (e.g., 50%)")
        p2 = st.number_input("Proportion in 2nd (Test) Group (%)", value=40.0, min_value=0.0, max_value=99.99, help="Enter a percentage value (e.g., 50%)")
        power = st.number_input("Power (%)", value=80.0, min_value=0.0, max_value=99.99, help="Enter a percentage value (e.g., 80%)")
        drpt = st.number_input("Drop-Out (%)", min_value=0.0, value=0.0, max_value=50.0, help="Enter a percentage value (e.g., 1%)")

        x = st.radio("Choose Method for Design Effect:", options=['Given', 'Calculate'])

        if x == "Given":
            designEffect = st.number_input("Design Effect (Given)", value=1.0, min_value=1.0, help="Enter a decimal value (e.g., 1.5)")
            m = None
            ICC = None
        else:
            m = st.number_input("Number of Clusters (m)", min_value=2, value=4, help="Enter an integer value (e.g., 4)")
            ICC = st.number_input("Intra-class Correlation (ICC) for clustering", min_value=0.0, max_value=1.0, value=0.05, help="Enter a decimal value (e.g., 0.05)")
            designEffect = 1 + (m - 1) * ICC
            st.metric("Design Effect", value=round(designEffect, 2))

        # ✅ Form submit button (replaces old "Calculate Sample Size" button)
        submitted = st.form_submit_button("Calculate Sample Size")

    if submitted:
        handle_go()  # sets session_state.go_clicked = True

    # Helper to generate label for dropdown
    def make_prop_test_history_label(p1, p2, power, drpt, designEffect, m=None, ICC=None, method="Given"):
        if method == "Given":
            return f"Ref={p1}%, Test={p2}%, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
        else:
            return (f"Ref={p1}%, Test={p2}%, Power={power}%, DropOut={drpt}%, "
                    f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.prop_test_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        prop_test_options = [make_prop_test_history_label(**entry) for entry in st.session_state.prop_test_history]
        selected_label = st.selectbox("Choose a past input set:", prop_test_options, key="prop_test_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.prop_test_history
                                    if make_prop_test_history_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History", key="submit_button", on_click=handle_submit)
        else:
            hist_submit = False
    else:
        hist_submit = False

    # Calculation condition
    if submitted or st.session_state.submit_clicked:
        if hist_submit and selected_history:
            # Use selected history
            p1 = selected_history["p1"]
            p2 = selected_history["p2"]
            power = selected_history["power"]
            drpt = selected_history["drpt"]
            designEffect = selected_history["designEffect"]
            delta = abs(p2 - p1)
        else:
            # Add current input to history
            new_entry = {
                "p1": p1,
                "p2": p2,
                "power": power,
                "drpt": drpt,
                "designEffect": designEffect,
                "m": m,
                "ICC": ICC,
                "method": x
            }
            st.session_state.prop_test_history.append(new_entry)

        confidenceIntervals = [0.95, 0.8, 0.9, 0.97, 0.99, 0.999, 0.9999]
        out = []

        for conf in confidenceIntervals:
            sample_size = nSampleProp(p1=(p1 / 100), p2=(p2 / 100), delta=(delta / 100),
                                    Pw=(power / 100), Conf=conf,
                                    designEf=designEffect, dropOut=(drpt / 100))
            out.append(sample_size)

        df = pd.DataFrame({
            "Confidence Levels (%)": [cl * 100 for cl in confidenceIntervals],
            "Sample Size": out
        })

        dds = nSampleProp(p1=(p1 / 100), p2=(p2 / 100), delta=(delta / 100),
                        Pw=(power / 100), Conf=0.95,
                        designEf=designEffect, dropOut=(drpt / 100))

        st.write("The study would need a total sample size of:")
        st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <div style="
                font-size: 36px;
                font-weight: bold;
                background-color: #48D1CC;
                padding: 10px;
                border-radius: 10px;
                text-align: center;">
                {2 * dds}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"""Participants (i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{dds}</span> individuals in each group) 
                    to achieve a power of {power}% and 
                    <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, 
                    for detecting a difference in proportions of {delta}% 
                    between the test and reference group, assuming that {p1}% of the subjects in the reference population have the factor of interest. 
                    Design effect is **{round(designEffect, 1)}** with **{drpt}%** drop-out.""",
                unsafe_allow_html=True)

        st.subheader("📊 Sample Sizes at Other Confidence Levels")
        st.dataframe(df)
        st.markdown("---") 
        # D efine power and confidence levels
        powers = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]
        conf_levels = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]

        st.subheader("📈 Sample Size Cross Table for Different Powers and Confidence Levels")

        power_labels = [f"{int(p * 100)}%" for p in powers]
        conf_labels = [f"{int(c * 100)}%" for c in conf_levels]
        cross_table = pd.DataFrame(index=conf_labels, columns=power_labels)
        # Fill the cross table
        for i, conf in enumerate(conf_levels):
            for j, power_val in enumerate(powers):
                ss = nSampleProp(
                    p1=(p1 / 100),
                    p2=(p2 / 100),
                    delta=(delta / 100),
                    Pw=power_val,
                    Conf=conf,
                    designEf=designEffect,
                    dropOut=(drpt / 100)
                )
                cross_table.iloc[i, j] = ss

        # Label table
        cross_table.index.name = "Confidence Level (%)"
        cross_table.columns.name = "Power (%)"

        st.dataframe(cross_table)
        st.write("**Rows are Confidence Levels; Columns are Powers**")
        st.session_state["cross_table"] = cross_table

    ## Plot
    if "cross_table" in st.session_state:
        cross_table = st.session_state.cross_table
        import matplotlib.pyplot as plt
        powers = [int(col.strip('%')) for col in cross_table.columns]
        conf_levels = [int(row.strip('%')) for row in cross_table.index]
        alpha_levels = [100 - c for c in conf_levels]

        # Let user select:
        conf_to_plot = st.selectbox("📌 Select Confidence Level for Power Plot (X-axis: Power)", cross_table.index, index=cross_table.index.tolist().index("95%"))
        power_to_plot = st.selectbox("📌 Select Power Level for Alpha Plot (X-axis: Confidence Level)", cross_table.columns, index=cross_table.columns.tolist().index("80%"))

        # Extract corresponding sample size rows/columns
        sample_sizes_power = cross_table.loc[conf_to_plot].astype(float).tolist()
        sample_sizes_alpha = cross_table[power_to_plot].astype(float).tolist()

        # Convert string labels to numbers
        selected_conf_level = int(conf_to_plot.strip('%'))
        selected_power = int(power_to_plot.strip('%'))

        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Power axis (primary y-axis)
        ax1.plot(sample_sizes_power, powers, marker='o', linestyle='-', color='blue', label=f'Power (%) at {conf_to_plot} CL')
        ax1.set_xlabel("Sample Size")
        ax1.set_ylabel("Power (%)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim([60, 100])

        # Alpha axis (secondary y-axis)
        ax2 = ax1.twinx()
        ax2.plot(sample_sizes_alpha, alpha_levels, marker='o', linestyle='-', color='orange', label=f'Alpha (%) at {power_to_plot} Power')
        ax2.set_ylabel("Alpha Level (%)", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim([0, 30])

        # Combined legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

        # Title, grid, layout
        plt.title("Sample Size vs Power and Alpha Level (from Cross Table)")
        plt.grid(True)
        plt.tight_layout()

        # Show in Streamlit
        st.pyplot(fig)
    # Footer and references
    st.markdown("---")
    st.subheader("📌 Formula for Sample Size Calculation")

    st.markdown("### **Two-Sample Proportion Hypothesis Test Sample Size Formula**")
    st.latex(r"""
    n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (p_1(1 - p_1) + p_2(1 - p_2))}{\delta^2} \times \frac{DE}{1 - \text{Dropout\%}}
    """)

    st.markdown("### **Design Effect Calculation (if clusters are used):**")
    st.latex(r"""
    DE = 1 + (m - 1) \times ICC
    """)

    st.subheader("📌 Description of Parameters")
    st.markdown("""
    - **\( Z_{1-alpha/2} \)**: Critical value for the confidence level (e.g., 1.96 for 95% confidence).
    - **\( Z_{1-beta} \)**: Power.
    - **\( p_1 \)**, **\( p_2 \)**: Proportions in the first and second groups.
    - **\( \delta \)**: Expected difference between the two proportions.
    - **\( DE \) (Design Effect)**: Adjusts for clustering in sample selection.
    - **\( m \)**: Number of cluster.
    - **\( ICC \)**: Intra-cluster correlation coefficient.
    - **Dropout%**: Anticipated percentage of dropout in the study.
    """)

    st.subheader("📌 References")
    st.markdown("""
    1. **Lemeshow, S., Hosmer Jr., D.W., Klar, J., Lwanga, S.K. (1990).** Adequacy of sample size in health studies. England: John Wiley & Sons.
    """)

    st.markdown("---")
    st.subheader("Citation")
    st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")
