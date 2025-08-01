import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm


def main():
    #    st.set_page_config(page_title="StudySizer | Poisson Rates (Person-Time) or Negative Binomial (Person-Time) |Over dispersion", page_icon="🧮")
    #
    #    st.markdown("""
    #    <style>
    #    button[data-testid="stBaseButton-header"] {
    #        display: none !important;
    #    }
    #    </style>
    #    """, unsafe_allow_html=True)


    chooseButton= st.sidebar.radio("Choose Method", options=["Help","Poisson Distribution (No over dispersion)", "Negative Binomial Distribution (Overdisperdion in Poisson varianve)"],index=0)
    if chooseButton=="Help":
        st.title("Approaches to Sample Size Calculation for Comparing Two Event Rates")
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
        st.markdown("""
        ## **Introduction**

        In medical and epidemiological studies, comparing event rates between two groups is a common objective. When the outcome is measured as a count of events over person-time (e.g., infection rates per 1,000 person-years), selecting an appropriate method for calculating the sample size becomes crucial. Several statistical approaches exist to guide this process, with two predominant methodologies: the **Generalized Linear Model (GLM)-based approach** and the **Normal approximation method**. This blog explores these methods, their assumptions, formulas, use cases, and how to account for overdispersion when needed.

        ---
        """)

        st.header("1. GLM-Based Sample Size Calculation")
        st.subheader("a. Theoretical Foundation")
        st.markdown("""
        The GLM-based method relies on the Poisson regression model, a special case of the GLM family, which models count data under the assumption that the **variance equals the mean** (Poisson distribution). For comparing two Poisson rates (e.g., treatment vs. control), we evaluate the difference in log rates:
        """)
        st.latex(r"H_0: \log(\lambda_1) = \log(\lambda_2) \quad \text{vs} \quad H_1: \log(\lambda_1) \neq \log(\lambda_2)")
        st.latex(r"\text{where},\lambda_1 \text{ and } \lambda_2 \text{are the rates in the two groups.}")

        st.subheader("b. Formula")
        st.markdown("The GLM-based sample size formula for comparing two Poisson rates over person-time is:")
        st.latex(r"""
        n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \left( \frac{1}{Q_1 \mu_1 T_1} + \frac{1}{Q_0 \mu_0 T_0} \right)}{[\log(\mu_1 T_1 / \mu_0 T_0)]^2} \cdot \frac{DE}{1 - \text{Dropout\%}}
        """)

        st.markdown("""
        - \( Z_{1-alpha/2} \): Z-value for confidence level (e.g., 1.96 for 95%)
        - \( Z_{1-beta} \): Z-value for power (e.g., 0.84 for 80%)
        - \( Q0, Q1 \): Proportion of total sample in control and treatment groups
        - \( mu0, mu1 \): Event rates in the two groups
        - \( T0, T1 \): Follow-up time
        - \( DE \): Design effect (for cluster sampling)
        - Dropout%: Anticipated dropout rate
        """)

        st.subheader("c. Parameter Selection Example")
        st.markdown("""
        Suppose:
        - Control group rate (mu_0) = 0.5 events/person-year  
        - Treatment group rate (mu_1) = 0.35  
        - Follow-up time = 1 year  
        - Power = 80%, Confidence Level = 95%  
        - Equal allocation ( Q0 = Q1 = 0.5 )  
        - No dropout, DE = 1  
        """)

        st.markdown("Then sample size per group:")
        st.latex(r"""
        n \approx \frac{(1.96 + 0.84)^2 \cdot \left(\frac{1}{0.5 \cdot 0.35} + \frac{1}{0.5 \cdot 0.5}\right)}{[\log(0.35 / 0.5)]^2} \approx 300 \text{ per group}
        """)

        st.header("2. Normal Approximation Method")
        st.markdown("""
        This approach approximates the distribution of the difference in rates using the central limit theorem. It assumes the **rate difference** is approximately normally distributed when sample size is large.
        """)

        st.subheader("a. Formula")
        st.latex(r"""
        n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot (\lambda_1 + \lambda_2)}{(\lambda_1 - \lambda_2)^2} \cdot T
        """)

        st.markdown("""
        This formula is simpler and historically used for rough sample size estimation. It doesn't easily account for unequal follow-up, cluster design, or overdispersion.
        """)

        st.subheader("b. Limitations")
        st.markdown("""
        - Valid only when count data is not highly skewed  
        - Assumes equal person-time per subject  
        - Doesn't handle dropout or design effect elegantly  
        """)

        st.header("3. Overdispersion and the Negative Binomial Model")
        st.markdown("""
        In real-world data, the assumption that variance equals mean (Poisson) often fails. Overdispersion (variance > mean) leads to underestimated standard errors and inflated Type I error.
        """)

        st.subheader("a. Negative Binomial Approach")
        st.markdown("The negative binomial distribution accounts for overdispersion by introducing a dispersion parameter \( k \), where:")
        st.latex(r"\text{Variance} = \mu + \mu^2/k")

        st.markdown("Sample size formula for negative binomial comparison:")
        st.latex(r"""
        n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \left( \frac{1}{Q_1}\left(\frac{1}{\mu_1} + \frac{1}{k_1}\right) + \frac{1}{Q_0}\left(\frac{1}{\mu_0} + \frac{1}{k_0}\right) \right)}{[\log(\mu_1 T_1 / \mu_0 T_0)]^2} \cdot \frac{DE}{1 - \text{Dropout\%}}
        """)

        st.subheader("b. Choosing \( k \)")
        st.markdown("""
        | k Value | Description              | Use Case                              |
        |---------|---------------------------|----------------------------------------|
        | ( infinity ) | No overdispersion         | Pure Poisson assumption                |
        | 10–20  | Mild overdispersion       | Moderately dispersed clinical data     |
        | 1–10   | Moderate overdispersion   | Most health/epidemiological studies    |
        | 0.1–1  | Strong overdispersion     | Hospital visits, infections            |
        | <0.1    | Extreme overdispersion    | Clustered, zero-inflated outcomes      |
        """, unsafe_allow_html=True)

        st.header("4. Comparison of Approaches")
        st.markdown("""
        | Feature                 | GLM-Poisson              | Normal Approximation      | Negative Binomial (GLM)   |
        |------------------------|--------------------------|----------------------------|----------------------------|
        | Handles Unequal Time   | Yes                      | No                         | Yes                        |
        | Cluster Design         | Yes (via DE)             | Limited                    | Yes                        |
        | Dropout Adjustment     | Yes                      | No                         | Yes                        |
        | Overdispersion Support | No                       | No                         | Yes                        |
        | Complexity             | Moderate                 | Low                        | High                       |
        """, unsafe_allow_html=True)

        st.header("Conclusion")
        st.markdown("""
        For accurate and robust sample size estimation in rate comparison studies, the **GLM approach** is preferred, especially when follow-up times vary or dropout is anticipated. However, **overdispersion** should be assessed, and the **negative binomial model** used when necessary. While the **normal approximation method** provides a simple alternative, it is increasingly being replaced due to its limitations in modern clinical research contexts.
        """)
        st.markdown("---")
        st.markdown("**Developed by [Rajesh Majumder]**")
        st.markdown("**Email:** rajeshnbp9051@gmail.com")
        st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")

    elif chooseButton=="Poisson Distribution (No over dispersion)":
        st.title("Sample Size Calculation for Comparing Two Poisson Rates (Person-Time) | H0: Both the group rates are same")
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
        
        # Function: Sample size for Poisson rate comparison (GLM approach with person-time, design effect, dropout)
        def nSamplePoissonGLM(mu0, mu1, T0=1.0, T1=1.0, Q0=0.5, Q1=0.5, alpha=0.05, power=0.8, designEffect=1.0, dropout=0.0):
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(power)
            lambda0 = mu0 * T0
            lambda1 = mu1 * T1
            if lambda0 == 0 or lambda1 == 0:
                return np.nan 
            logRR = np.log(lambda0 / lambda1)
            if logRR == 0:
                return np.nan 
            variance_term = (1 / (Q1 * lambda1)) + (1 / (Q0 * lambda0))
            sqrt_N = (z_alpha + z_beta) * np.sqrt(variance_term) / abs(logRR)
            N = sqrt_N ** 2
            N_adj = N * designEffect / (1 - dropout)
            return np.ceil(N_adj)

        # Initialize history store
        if "poisson_rate_history" not in st.session_state:
            st.session_state.poisson_rate_history = []

        # Sidebar inputs
        st.sidebar.markdown("---")
        st.sidebar.header("🔧 Input Parameters")
        mu0 = st.sidebar.number_input("Event Rate in Control Group (% per person-time)", value=50.0, min_value=0.1,max_value=99.99,help="Enter a percentage value (e.g., 50%)")/100.0
        mu1 = st.sidebar.number_input("Event Rate in Treatment Group (% per person-time)", value=35.0, min_value=0.01,max_value=99.99,help="Enter a percentage value (e.g., 35%)")/100.0
        T0 = st.sidebar.number_input("Follow-up Time (T0) (Control)", value=1.0, min_value=0.01,help="Enter any positive value (e.g., 1= 1 Year, 2= 2 Years, 0.5= 6 months)")
        T1 = st.sidebar.number_input("Follow-up Time (T1) (Treatment)", value=1.0, min_value=0.01,help="Enter any positive value (e.g., 1= 1 Year, 2= 2 Years, 0.5= 6 months)")

        design=st.sidebar.radio("Choose Group proportion",["Equal proportion", "Unequal proportion"])
        st.sidebar.text("Choose Unequal proportion when the study groups are not equally sampled.")
        if design=="Equal proportion":
            Q0=0.5
            Q1=0.5
            col1,col2=st.columns(2)
            col1.metric("Control Group Q0",value=Q0)
            col2.metric("Treatement Group Q1",value=Q1)
        else:
            Q0=st.sidebar._number_input("Enter the Control group proportion (Q0) (%)",min_value=0.0, max_value=99.99, value=55.0,help="Enter a percentage value (e.g., 55%)")/100
            Q1=1-Q0
            col1,col2=st.columns(2)
            col1.metric("Control Group Q0",value=round(Q0,2))
            col2.metric("Treatement Group Q1",value=round(Q1,2))

        power = st.sidebar.number_input("Power (%)",  min_value=50.0, max_value=99.9, value=80.0,help="Enter a percentage value (e.g., 80%)") / 100
        drpt = st.sidebar.number_input("Drop-Out (%)", value=0.0, max_value= 50.0, min_value=0.0,help="Enter a percentage value (e.g., 1%)") / 100

        # Design Effect
        design_method = st.sidebar.radio("Design Effect Option:", ["Given", "Calculate"])
        if design_method == "Given":
            designEffect = st.sidebar.number_input("Design Effect (Given)", value=1.0, min_value=1.0,help="Enter an decimal value (e.g., 1.5)")
            m=None
            ICC=None
        else:
            m = st.sidebar.number_input("Number of Clusters (m)", min_value=2,value=4,help="Enter an integer value (e.g., 4)")
            ICC = st.sidebar.number_input("Intra-cluster Correlation (ICC) for clustering", min_value=0.0, max_value=1.0, value=0.05,help="Enter a decimal value (e.g., 0.05)")
            designEffect = 1 + (m - 1) * ICC
            col1,col2,col3=st.columns(3)
            col1.metric("Cluster Size (m)",value=m)
            col2.metric("Intra Class Correlation (ICC)",value=ICC)
            col3.metric("Design Effect",value= round(designEffect,2))
        # Calculate button
        go = st.button("Calculate Sample Size")

        # Helper to generate label for dropdown
        def make_poisson_rate_history_label(mu0, mu1, T0, T1, Q0, Q1, power, drpt, designEffect, m=None, ICC=None, method="Given"):
            if method == "Given":
                return f"mu0={mu0}, mu1={mu1},T0={T0}, T1={T1}, Q0={Q0}, Q1={Q1}, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
            else:
                return (f"mu0={mu0}, mu1={mu1},T0={T0}, T1={T1}, Q0={Q0}, Q1={Q1}, Power={power}%, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

        # Select from history
        selected_history = None
        selected_label = None

        if st.session_state.poisson_rate_history:
            st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
            poisson_rate_options = [make_poisson_rate_history_label(**entry) for entry in st.session_state.poisson_rate_history]
            selected_label = st.selectbox("Choose a past input set:", poisson_rate_options, key="poisson_rate_history_selector")

            if selected_label:
                selected_history = next((item for item in st.session_state.poisson_rate_history
                                        if make_poisson_rate_history_label(**item) == selected_label), None)
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
                    mu0= selected_history["mu0"]
                    mu1= selected_history["mu1"]
                    T0= selected_history["T0"]
                    T1= selected_history["T1"]
                    power = selected_history["power"]
                    drpt = selected_history["drpt"]
                    designEffect = selected_history["designEffect"]
                    Q1= selected_history["Q1"]
                    Q0= selected_history["Q0"]
                else:
                    # Add current input to history
                    new_entry = {
                        "mu0":mu0,
                        "mu1":mu1,
                        "T0":T0,
                        "T1":T1,
                        "power":power,
                        "drpt":drpt,
                        "designEffect":designEffect,
                        "m":m,
                        "ICC":ICC,
                        "method":design_method,
                        "Q1":Q1,
                        "Q0":Q0
                    }
                    st.session_state.poisson_rate_history.append(new_entry)

                confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
                out=[]

                for conf in confidenceIntervals:
                    sample_size= nSamplePoissonGLM(mu0=mu0, mu1=mu1, T0=T0, T1=T1,Q0=Q0, Q1=Q1, alpha=(1-(conf)), power=power, designEffect=designEffect, dropout=drpt)
                    out.append(sample_size)

                df= pd.DataFrame({
                    "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
                    "Sample Size": out
                })

                dds= nSamplePoissonGLM(mu0=mu0, mu1=mu1, T0=T0, T1=T1,Q0=Q0, Q1=Q1, alpha=0.05,designEffect=designEffect,dropout=drpt)

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
                        {int(dds)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f""" number of individuals (i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds*Q0)}</span> and <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds*Q1)}</span> individuals respectively in control and intervention group with unequal Sample size ratio= {round(Q0,2)}:{round(Q1,2)} % respectively) to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, for detecting between the control and treatment groups in event rates of {mu0} and {mu1} per person-time respectively for control and intervention groups. The calculation is base on the assumeption that the population variance is equal to the population mean and also considered that, the follow-up time as {T0} and {T1} respectively for control and Intervention groups, with the design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out rate.""",unsafe_allow_html=True)
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
                        ss = nSamplePoissonGLM(mu0=mu0, mu1=mu1, T0=T0, T1=T1,Q0=Q0, Q1=Q1, alpha=(1-(conf)), power=power_val, designEffect=designEffect, dropout=drpt) 
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
            st.markdown("### **Sample Size Formula for Comparing Two Poisson Rates (Person-Time)**")

            st.latex(r"""
                n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \left( \frac{1}{Q_1 \mu_1 T_1} + \frac{1}{Q_0 \mu_0 T_0} \right)}{\left[ \log \left( \frac{\mu_1 T_1}{\mu_0 T_0} \right) \right]^2} \times \frac{DE}{1 - \text{Dropout\%}}
            """)
            st.markdown("### **Design Effect Calculation (if clusters are used):**")
            st.latex(r"""
            DE = 1 + (m - 1) \times ICC
            """)

            st.subheader("📌 Description of Parameters")

            st.markdown("""
            - **\( Z_{1-alpha/2} \)**: Z-value corresponding to desired confidence level.
            - **\( Z_{1-beta}\)**: Z-value corresponding to desired power.
            - **\(mu0\)**: Event rate in the control group per person-time.
            - **\(mu1 \)**: Event rate in the treatment group per person-time.
            - **\( T0 \)**: Follow-up time for the control group.
                - (e.g, T0=1 : per year | T0=0.5 : 6 months follow-up | T0=2.0 : Each subject if followed for 2 years)
            - **\( T1 \)**: Follow-up time for the treatment group.
                - (e.g, same as T0)
            - **\( Q0 \)**: Proportion of participants in the control group.
            - **\( Q1 \)**: Proportion of participants in the treatment group.
            - **\( DE \)**: Design Effect to adjust for cluster sampling (if applicable).
            - **\( m \)**: Average number of subjects per cluster.
            - **\( ICC \)**: Intra-class correlation coefficient.
            - **Dropout%**: Anticipated percentage of dropout in the study.
            """)

            st.subheader("📌 References")

            st.markdown("""
            1. Cundill, Bonnie, and Neal D E Alexander. “Sample size calculations for skewed distributions.” BMC medical research methodology vol. 15 28. 2 Apr. 2015, doi:10.1186/s12874-015-0023-0
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
    else:
        st.title("Sample Size Calculation for Comparing Two Negative Binomial Rates (Person-Time) | H0: Both the group rates are same")
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
        
        # Function: Sample size for Negative Binomial rate comparison (GLM approach with person-time, design effect, dropout)
        def nSampleNegBinGLM(mu0, mu1, k0, k1, T0=1.0, T1=1.0, Q0=0.5, Q1=0.5, alpha=0.05, power=0.8, designEffect=1.0, dropout=0.0):
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(power)
            lambda0 = mu0 * T0
            lambda1 = mu1 * T1
            if lambda0 == 0 or lambda1 == 0:
                return np.nan 
            logRR = np.log(lambda0 / lambda1)
            if logRR == 0:
                return np.nan 
            var_term = (1 / Q1) * (1 / mu1 + 1 / k1) + (1 / Q0) * (1 / mu0 + 1 / k0)
            sqrt_N = (z_alpha + z_beta) * np.sqrt(var_term) / abs(logRR)
            N = sqrt_N ** 2
            N_adj = N * designEffect / (1 - dropout)
            return np.ceil(N_adj)

        # Initialize history store
        if "poisson_rate_overdisp_history" not in st.session_state:
            st.session_state.poisson_rate_overdisp_history = []

        st.sidebar.markdown("---")

        st.sidebar.header("🔧 Input Parameters")

        # Sidebar inputs
        mu0 = st.sidebar.number_input("Event Rate in Control Group (% per person-time)", value=50.0, min_value=0.1, max_value=99.99,help="Enter a percentage value (e.g., 50%)") / 100
        mu1 = st.sidebar.number_input("Event Rate in Treatment Group (% per person-time)", value=35.0, min_value=0.1, max_value=99.99,help="Enter a percentage value (e.g., 35%)") / 100
        K0 = st.sidebar.number_input("Dispersion (k) for Control Group", min_value=0.01, value=10.0,help="Enter a any value >0")
        K1 = st.sidebar.number_input("Dispersion (k) for Treatment Group", min_value=0.01, value=10.0,help="Enter a any value >0")
        T0 = st.sidebar.number_input("Follow-up Time (T0) (Control)", value=1.0, min_value=0.01,help="Enter any positive value (e.g., 1= 1 Year, 2= 2 Years, 0.5= 6 months)")
        T1 = st.sidebar.number_input("Follow-up Time (T1) (Treatment)", value=1.0, min_value=0.01,help="Enter any positive value (e.g., 1= 1 Year, 2= 2 Years, 0.5= 6 months)")

        design = st.sidebar.radio("Choose Group proportion", ["Equal proportion", "Unequal proportion"])
        if design == "Equal proportion":
            Q0 = 0.5
            Q1 = 0.5
            col1,col2=st.columns(2)
            col1.metric("Control Group Q0",value=round(Q0,2))
            col2.metric("Treatement Group Q1",value=round(Q1,2))
        else:
            Q0 = st.sidebar.number_input("Control Group Proportion (Q0 %)", min_value=1.0, max_value=99.0, value=50.0,help="Enter a percentage value (e.g., 50%)") / 100
            Q1 = 1 - Q0
            col1,col2=st.columns(2)
            col1.metric("Control Group Q0",value=round(Q0,2))
            col2.metric("Treatement Group Q1",value=round(Q1,2))

        power = st.sidebar.number_input("Power (%)", min_value=50.0, max_value=99.9, value=80.0,help="Enter a percentage value (e.g., 80%)") / 100
        drpt = st.sidebar.number_input("Dropout Rate (%)", value=0.0, max_value=50.0, min_value=0.0,help="Enter a percentage value (e.g., 1%)") / 100

        design_method = st.sidebar.radio("Design Effect Option:", ["Given", "Calculate"])
        if design_method == "Given":
            designEffect = st.sidebar.number_input("Design Effect (Given)", value=1.0, min_value=1.0,help= "Enter an decimal value (e.g., 1.5)")
            m = None
            ICC = None
        else:
            m = st.sidebar.number_input("Number of Clusters (m)", min_value=2,value=4,help="Enter an integer value (e.g., 4)")
            ICC = st.sidebar.number_input("Intra-class Correlation (ICC) for clustering", min_value=0.0, max_value=1.0, value=0.05,help="Enter a decimal value (e.g., 0.05)")
            designEffect = 1 + (m - 1) * ICC
            col1,col2,col3=st.columns(3)
            col1.metric("Cluster Size (m)",value=m)
            col2.metric("Intra Class Correlation (ICC)",value=ICC)
            col3.metric("Design Effect",value= round(designEffect,2))

        # Calculate button
        go = st.button("Calculate Sample Size")


        # Helper to generate label for dropdown
        def make_poisson_rate_overdisp_history_label(mu0, mu1, T0, T1, K0, K1, Q0, Q1, power, drpt, designEffect, m=None, ICC=None, method="Given"):
            if method == "Given":
                return f"mu0={mu0}, mu1={mu1},T0={T0}, T1={T1}, K0={K0}, K1={K1}, Q0={Q0}, Q1={Q1}, Power={power}%, DropOut={drpt}%, DE(Given)={round(designEffect, 2)}"
            else:
                return (f"mu0={mu0}, mu1={mu1},T0={T0}, T1={T1}, K0={K0}, K1={K1}, Q0={Q0}, Q1={Q1}, Power={power}%, DropOut={drpt}%, "
                        f"DE(Calc)={round(designEffect, 2)}, m={m}, ICC={ICC}")

        # Select from history
        selected_history = None
        selected_label = None

        if st.session_state.poisson_rate_overdisp_history:
            st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
            poisson_rate_overdisp_options = [make_poisson_rate_overdisp_history_label(**entry) for entry in st.session_state.poisson_rate_overdisp_history]
            selected_label = st.selectbox("Choose a past input set:", poisson_rate_overdisp_options, key="poisson_rate_overdisp_history_selector")

            if selected_label:
                selected_history = next((item for item in st.session_state.poisson_rate_overdisp_history
                                        if make_poisson_rate_overdisp_history_label(**item) == selected_label), None)
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
                    mu0= selected_history["mu0"]
                    mu1= selected_history["mu1"]
                    T0= selected_history["T0"]
                    T1= selected_history["T1"]
                    K0= selected_history["K0"]
                    K1= selected_history["K1"]
                    power = selected_history["power"]
                    drpt = selected_history["drpt"]
                    designEffect = selected_history["designEffect"]
                    Q1= selected_history["Q1"]
                    Q0= selected_history["Q0"]
                else:
                    # Add current input to history
                    new_entry = {
                        "mu0":mu0,
                        "mu1":mu1,
                        "T0":T0,
                        "T1":T1,
                        "K0":K1,
                        "K1":K1,
                        "power":power,
                        "drpt":drpt,
                        "designEffect":designEffect,
                        "m":m,
                        "ICC":ICC,
                        "method":design_method,
                        "Q1":Q1,
                        "Q0":Q0
                    }
                    st.session_state.poisson_rate_overdisp_history.append(new_entry)


                confidenceIntervals = [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
                out = []
                for conf in confidenceIntervals:
                    sample_size = nSampleNegBinGLM(mu0=mu0, mu1=mu1,k0=K0, k1=K1, T0=T0, T1=T1, Q0=Q0, Q1=Q1, alpha=(1 - conf), power=power, designEffect=designEffect, dropout=drpt)
                    out.append(sample_size)

                df = pd.DataFrame({
                    "Confidence Levels (%)": [round(c * 100,2) for c in confidenceIntervals],
                    "Total Sample Size": out
                })

                dds = nSampleNegBinGLM(mu0=mu0, mu1=mu1,k0=K0, k1=K1, T0=T0, T1=T1, Q0=Q0, Q1=Q1, alpha=0.05, power=power, designEffect=designEffect, dropout=drpt)

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
                        {int(dds)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f""" number of individuals (i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds*Q0)}</span> and <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds*Q1)}</span> individuals respectively in control and intervention group with unequal Sample size ratio= {round(Q0*100.0,2)}:{round(Q1*100,2)} % respectively) to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, for detecting between the control and treatment groups in event rates of {mu0} and {mu1} per person-time respectively for control and intervention groups. The calculation is based on the assumeption that both the groups are overly dispersed with over dispersion parameters K0={K0} and K1={K1} respectively also considered that, the follow-up time as {T0} and {T1} respectively for control and Intervention groups, with the design effect of **{round(designEffect,1)}** and **{(drpt)}%** drop-out rate.""",unsafe_allow_html=True)
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
                        ss = nSampleNegBinGLM(mu0=mu0, mu1=mu1,k0=K0, k1=K1, T0=T0, T1=T1, Q0=Q0, Q1=Q1, alpha=(1 - conf), power=power_val, designEffect=designEffect, dropout=drpt)
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
            st.markdown("### **GLM-based Sample Size Formula for Comparing Two Poisson Rates (Person-Time)- at overdispersion | Negative Binomial Rates**")
            st.latex(r"""
                n = \frac{(Z_{1-\alpha/2} + Z_{1-\beta})^2 \cdot \left( \frac{1}{Q_1} \left(\frac{1}{\mu_1} + \frac{1}{k_1}\right) + \frac{1}{Q_0} \left(\frac{1}{\mu_0} + \frac{1}{k_0}\right) \right)}{\left[\log\left(\frac{\mu_1 T_1}{\mu_0 T_0}\right)\right]^2} \cdot \frac{DE}{1 - \text{Dropout\%}}
            """)

            st.markdown("### **Design Effect (if cluster sampling):**")
            st.latex(r"""
            DE = 1 + (m - 1) \times ICC
            """)

            st.subheader("📌 Description of Parameters")

            st.markdown("""
            - **\( Z_{1-alpha/2} \)**: Z-value corresponding to desired confidence level.
            - **\( Z_{1-beta}\)**: Z-value corresponding to desired power.
            - **\(mu0\)**: Event rate in the control group per person-time.
            - **\(mu1 \)**: Event rate in the treatment group per person-time.
            - **\( T0 \)**: Follow-up time for the control group.
                -  (e.g, T0=1 : per year | T0=0.5 : 6 months follow-up | T0=2.0 : Each subject if followed for 2 years)
            - **\( T1 \)**: Follow-up time for the treatment group.
                -  (e.g, same as T0)
            - **\(K0\)**: Control group dispersion parameter.
            - **\(K1\)**: Treatment group dispersion parameter.
            - **\( Q0 \)**: Proportion of participants in the control group.
            - **\( Q1 \)**: Proportion of participants in the treatment group.
            - **\( DE \)**: Design Effect to adjust for cluster sampling (if applicable).
            - **\( m \)**: Average number of subjects per cluster.
            - **\( ICC \)**: Intra-class correlation coefficient.
            - **Dropout%**: Anticipated percentage of dropout in the study.
            """)
            st.markdown("---")
            dispersion_data = {
                "k Value": ["∞", "10–20", "1–10", "0.1–1", "<0.1"],
                "Description": [
                    "No overdispersion",
                    "Mild overdispersion",
                    "Moderate overdispersion",
                    "Strong overdispersion",
                    "Extreme overdispersion"
                ],
                "Use Case or Interpretation": [
                    "Reduces to Poisson",
                    "Often seen in moderately dispersed data",
                    "Many real-world health/clinical datasets",
                    "Epidemiological data, hospital visits, etc.",
                    "Clustered, zero-inflated or highly skewed data"
                ]
            }

            df_dispersion = pd.DataFrame(dispersion_data)

        st.subheader(" 📌Dispersion Parameter (k) Guidelines")
        st.table(df_dispersion) 
        st.markdown("---")
        st.subheader("📌 References")
        st.markdown("""
        1. Cundill, Bonnie, and Neal D E Alexander. “Sample size calculations for skewed distributions.” BMC medical research methodology vol. 15 28. 2 Apr. 2015, doi:10.1186/s12874-015-0023-0
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