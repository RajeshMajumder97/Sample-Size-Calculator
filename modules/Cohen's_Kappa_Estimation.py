import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

def main():
    # Sample size calculation functions
    def sample_size_kappa_two_raters(kappa, p1, p2, se):
        numerator = (1 - kappa) * (
            4 * p1 * p2 * (1 - p1) * (1 - p2) * (1 + kappa) +
            (1 - 2 * p1) * (1 - 2 * p2) * (p1 * (1 - p2) + p2 * (1 - p1)) * kappa * (2 - kappa)
        )
        denominator = (se * (p1 * (1 - p2) + p2 * (1 - p1))) ** 2
        return numerator / denominator

    def sample_size_kappa_multi_raters(kappa, p, m, se):
        part1 = 2 / (m * (m - 1))
        part2 = (3 - 1 / (p * (1 - p))) * kappa
        part3 = (m - 1) * (4 - 1 / (p * (1 - p))) * (kappa ** 2) / m
        numerator = (1 - kappa) * (part1 - part2 + part3)
        return numerator / (se ** 2)

    def calc_kappa_ss(kappa, se):
        return (1.96 / se) ** 2 * (1 - kappa) ** 2

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Streamlit App
    st.title("Sample Size Calculation for Inter-Rater Agreement (Kappa Statistic)")

    st.markdown("""
    <style>
    button[data-testid="stBaseButton-header"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("🔧 Input Parameters")

    # Store history
    if "kappa_history" not in st.session_state:
        st.session_state.kappa_history = []

    method = st.sidebar.radio("Kappa Type:", options=["Two Raters", "Multiple Raters"])
    kappa = st.sidebar.number_input("Anticipated Kappa (κ)", value=0.7, min_value=0.0, max_value=1.0 ,help="Enter a decimal value (e.g., 0.07)")
    se_method = st.sidebar.radio("Standard Error Input Method", ["Give SE directly", "Calculate from CI bounds"])

    if se_method == "Give SE directly":
        se = st.sidebar.number_input("Standard Error (SE)", value=0.04, min_value=0.0001, max_value= 0.2,help="Enter a decimal value (e.g., 0.04)" )
        ci=None
        upper=None
        lower=None
    else:
        ci = st.sidebar.text_input("Confidence Interval (Lower, Upper) (comma-separated)", value="0.62,0.78")
        try:
            lower, upper = [float(x.strip()) for x in ci.split(",")]
            se = (upper - lower) / 3.92
        except:
            st.sidebar.warning("Enter the confidence interval values correctly.")
            st.stop()

    if method == "Two Raters":
        p1 = st.sidebar.number_input("Rater 1: Positive Proportion (p₁)", value=70.0, min_value=0.0, max_value=99.99,help="Enter a percentage value (e.g., 70%)") / 100
        p2 = st.sidebar.number_input("Rater 2: Positive Proportion (p₂)", value=70.0, min_value=0.0, max_value=99.99,help="Enter a percentage value (e.g., 70%)") / 100
        p, m = None, None
    else:
        p = st.sidebar.number_input("Overall Positive Proportion (p)", value=70.0, min_value=0.0, max_value=99.99,help="Enter a percentage value (e.g., 70%)") / 100
        m = st.sidebar.number_input("Number of Raters (m)", value=3, min_value=2,help="Enter an integer value (e.g., 3)")
        p1, p2 = None, None

    dropout = st.sidebar.number_input("Dropout (%)", value=0.0, min_value=0.0, max_value=50.0,help="Enter a percentage value (e.g., 1%)") / 100
    design_type = st.sidebar.radio("Design Effect:", ["Given", "Calculate"])

    if design_type == "Given":
        design_effect = st.sidebar.number_input("Design Effect (Given)", value=1.0,min_value=1.0,help= "Enter a decimal value (e.g., 1.5)")
        m_clust, ICC = None, None
    else:
        m_clust = st.sidebar.number_input("Number of Clusters (m)",min_value=2,value=4,help="Enter an integer value (e.g., 4)")
        ICC = st.sidebar.number_input("Intra-class Correlation (ICC) for clustering",min_value=0.0,max_value=1.0,value=0.05,help="Enter a decimal value (e.g., 0.05)")
        design_effect = 1 + (m_clust - 1) * ICC
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster Size (m)", value=m_clust)
        col2.metric("Intra Class Correlation (ICC)", value=ICC)
        col3.metric("Design Effect", value=round(design_effect, 2))

    go = st.button("Calculate Sample Size")

    # History label helper
    def make_kappa_label(method, kappa, se_method,se, lower, upper, p, p1, p2, m, dropout, design_type,design_effect, m_clust, ICC):
        return f"Kappa.Type={method}, Anticip.Kappa={kappa}, StandardError.Method={se_method},lower={lower},upper={upper}, SE={round(se,2)}, p1={p1}%, p2={p2}%, p={p}%, No.Raters= {m},  DropOut={dropout}%,Design.Type={design_type}, DE={round(design_effect, 2)}, No.Clusters={m_clust}, ICC={ICC}"
 
    # History selector
    # Select from history
    selected_history = None
    selected_label = None

    if st.session_state.kappa_history:
        st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
        kappa_options = [make_kappa_label(**entry) for entry in st.session_state.kappa_history]
        selected_label = st.selectbox("Choose a past input set:", kappa_options, key="kappa_history_selector")

        if selected_label:
            selected_history = next((item for item in st.session_state.kappa_history
                                    if make_kappa_label(**item) == selected_label), None)
            hist_submit = st.button("🔁 Recalculate from Selected History")
        else:
            hist_submit = False
    else:
        hist_submit = False

#method, kappa, se_method,se, ci, p, p1, p2, m, dropout, design_type,design_effect, m_clust, ICC
    if go or hist_submit:
        if hist_submit and selected_history:
            # Use selected history
            method= selected_history["method"]
            kappa= selected_history["kappa"]
            se_method = selected_history["se_method"]
            se = selected_history["se"]
            #ci = selected_history["ci"]
            lower = selected_history["lower"]
            upper = selected_history["upper"]
            p = selected_history["p"]
            p1 = selected_history["p1"]
            p2 =  selected_history["p2"]
            m =  selected_history["m"]
            dropout =  selected_history["dropout"]
            design_type =  selected_history["design_type"]
            design_effect =  selected_history["design_effect"]
            m_clust =  selected_history["m_clust"]
            ICC =  selected_history["ICC"]
        else:
            # Add current input to history
            new_entry = {
            "method":method,
            "kappa":kappa,
            "se_method":se_method,
            "se":se,
            #"ci":ci,
            "lower":lower,
            "upper":upper,
            "p":p,
            "p1":p1,
            "p2":p2,
            "m":m,
            "dropout":dropout,
            "design_type":design_type,
            "design_effect":design_effect,
            "m_clust":m_clust,
            "ICC":ICC
            }
            st.session_state.kappa_history.append(new_entry)

        if method == "Two Raters":
            if se_method == "Give SE directly":
                sample_size= sample_size_kappa_two_raters(kappa, p1, p2, se)
                sample_size = sample_size * design_effect / (1 - dropout)
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
                        {int(round(sample_size))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"""for estimating anticipeted kappa **{round(kappa,2)}** with SE {se} and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, by considering the positve proportions given by Rater 1 & Rater 2 are {p1}%, {p2}% respectively with design effect of **{round(design_effect,1)}** and **{(dropout)}%** drop-out from the sample.""",unsafe_allow_html=True)
            else:
                confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
                out=[]
                but=[]
                for conf in confidenceIntervals:
                    se= (upper - lower) / (2*norm.ppf(1-(1-conf)/2))
                    sample_size= sample_size_kappa_two_raters(kappa, p1, p2, se)
                    sample_size = sample_size * design_effect / (1 - dropout)
                    out.append(int(round(sample_size)))
                    but.append(se)
                df= pd.DataFrame({
                    "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
                    "Calculated SE": but,
                    "Sample Size": out
                })
                st.subheader("List of Sample Sizes at different Confidence Levels")
                st.dataframe(df)
                st.subheader("Assumptions:")
                st.write(f"These sample sizes are adjuster for design effect {round(design_effect,1)} and {dropout}% droupout to estimate anticipated kappa (k)= {kappa} (95% CI= [{ci}]) with positive proportions {p1}% {p2}% by Rater 1 and Rater 2 respectively.", unsafe_allow_html=True)
        else:
            if se_method == "Give SE directly":
                sample_size= sample_size_kappa_multi_raters(kappa, p, m, se)
                sample_size = sample_size * design_effect / (1 - dropout)
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
                        {int(round(sample_size))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"""for estimating anticipeted kappa **{round(kappa,2)}** with SE {se} and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, by considering the overall positve proportions is {p}% across all {m} Raters with design effect of **{round(design_effect,1)}** and **{(dropout)}%** drop-out from the sample.""",unsafe_allow_html=True)
            else:
                confidenceIntervals= [0.95,0.8,0.9,0.97,0.99,0.999,0.9999]
                out=[]
                but=[]
                for conf in confidenceIntervals:
                    se= (upper - lower) / (2*norm.ppf(1-(1-conf)/2))
                    sample_size= sample_size_kappa_multi_raters(kappa, p, m, se)
                    sample_size = sample_size * design_effect / (1 - dropout)
                    out.append(int(round(sample_size)))
                    but.append(se)
                df= pd.DataFrame({
                    "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
                    "Calculated SE": but,
                    "Sample Size": out
                })
                st.subheader("List of Sample Sizes at different Confidence Levels")
                st.dataframe(df)
                st.subheader("Assumptions:")
                st.write(f"These sample sizes are adjuster for design effect {round(design_effect,1)} and {dropout}% droupout to estimate anticipated kappa (k)= {kappa} (95% CI= [{ci}]) with overall positive proportions {p}% across all {m} Raters.", unsafe_allow_html=True)

    # Display formula
    st.markdown("---")
    with st.expander("Show the formula and the references"):
        st.subheader("📌 Formula for Sample Size Calculation")

        st.markdown("### 📌 **Case 1: Two Raters (m = 2)**")
        st.latex(r"""
        n = \frac{(1 - \kappa) \left[ 4p_1p_2(1 - p_1)(1 - p_2)(1 + \kappa) + (1 - 2p_1)(1 - 2p_2)(p_1(1 - p_2) + p_2(1 - p_1))\kappa(2 - \kappa) \right]}{\left[ SE \cdot (p_1(1 - p_2) + p_2(1 - p_1)) \right]^2}
        """)

        st.markdown("Where:")
        st.markdown("""
        - \( \kappa \): Anticipated Kappa agreement  
        - \( SE \): Desired standard error of Kappa  
        - \( p_1, p_2 \): Proportion of positive ratings by Rater 1 and Rater 2  
        - This formula is derived from the asymptotic variance of Kappa under independent ratings.
        """)

        st.markdown("### 📌 **Case 2: Multiple Raters (m > 2)**")
        st.latex(r"""
        n = \frac{(1 - \kappa) \left[ \frac{2}{m(m - 1)} - \left( 3 - \frac{1}{p(1 - p)} \right)\kappa + \frac{(m - 1)}{m} \left( 4 - \frac{1}{p(1 - p)} \right)\kappa^2 \right]}{SE^2}
        """)

        st.markdown("Where:")
        st.markdown("""
        - \( m \): Number of raters  
        - \( p \): Overall prevalence of the positive outcome  
        - \( SE \): Desired standard error of Kappa  
        - This generalizes Cohen’s Kappa to multiple raters using Fleiss’ method.
        """)

        st.markdown("### 📌 **Adjustment for Design Effect and Dropout**")
        st.latex(r"""
        n_{\text{adjusted}} = \frac{n \times DE}{1 - \text{dropout rate}}
        """)


        # Notes and developer info (always visible)
        st.markdown("---")
        st.subheader("📌 Notes")
        st.markdown("- This calculator uses the **normal approximation** to estimate required sample size.")
        st.markdown("- If insted of SE, confidence interval is given, it can be calculated as:")
        st.latex(r"""
                SE= \frac{(UCL-LCL)}{2 \times Z_{1-\frac{\alpha}{2}}}
                """)

        st.markdown("---")
        st.subheader("📌 References")
        st.markdown("""
        **Shoukri, M. M., Asyali, M. H., Donner, A. (2004).** Sample size requirements for the design of reliability study: review and new results. Statistical Methods in Medical Research, 13, 1-21.
        """)

    st.markdown("---")
    st.subheader("Citation")
    st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app](https://rajeshmajumderblog.netlify.app)")