import streamlit as st

def main():
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

    st.title("📖 Sample Size FAQs")
    st.markdown("Use this page to explore answers to common questions related to sample size calculation.")

    # ------------------------
    st.header("🔹 General Questions")

    with st.expander("❓ What is sample size and why is it important?"):
        st.markdown("Sample size is the number of subjects or units included in a study. It ensures the study has enough power to detect meaningful effects, avoids wasted resources, and ensures statistically valid results.")

    with st.expander("❓ What is statistical power?"):
        st.markdown("Power is the probability of detecting a true effect. Commonly set at 80% or 90%. A study with low power is more likely to miss real effects (Type II error).\n\nHigher power requires a larger sample size but increases the likelihood of identifying real associations. Use standard values like 80% or 90% unless there's a specific reason for more/less.")

    with st.expander("❓ What is the confidence level?"):
        st.markdown("The confidence level represents the certainty of your estimate. A 95% confidence level means if the study is repeated 100 times, the result would fall within the interval 95 times.\n\nIn sample size calculation, a 95% confidence level is usually paired with an alpha (significance level) of 0.05.")

    with st.expander("❓ What is design effect and when should it be used?"):
        st.markdown("Design effect adjusts sample size when data is clustered. It accounts for intra-cluster correlation (ICC) and increases sample size to maintain precision.\n\nIt’s commonly used in survey and public health research where people are grouped by location, hospital, or practice. When there is no clustering then design effect is 1.")

    with st.expander("❓ What is the difference between absolute and relative precision?"):
        st.markdown("Absolute precision is the fixed margin of error (e.g., ±5). Relative precision expresses error as a percentage of the estimate (e.g., 10% of the mean).\n\nUse absolute precision for fixed targets; use relative precision when proportions or estimates may vary.")

    # ------------------------
    st.header("🔎 Choosing the Right Sample Size Method")

    st.markdown("""
    Here's a quick guide to help you choose the correct sample size method based on your study design:

    | Scenario | Use When | Measure | Application |
    |---------|-----------|----------|-------------|
    | **Correlation Test** | Studying the strength and direction of relationship between two continuous variables | Pearson r | Observational studies |
    | **Intraclass Correlation (ICC)** | Measuring agreement within raters, clusters | ICC | Reliability studies, cluster sampling |
    | **Linear Regression** | One or more predictors influencing a continuous outcome | Regression coefficient | Predictive modeling, public health determinants |
    | **Logistic Regression** | Binary outcome (yes/no, disease/no disease) | Odds Ratio (OR) | Case-control or risk modeling |
    | **Proportion Estimation** | Estimating a single proportion | Expected proportion, margin of error | Prevalence studies |
    | **Mean Estimation** | Estimating a continuous variable | Expected mean and SD | Nutritional or clinical measurement studies |
    | **Two-Sample Comparison** | Comparing two groups (means or proportions) | Difference, SD or proportions | Clinical trials, intervention studies |
    | **Kappa Statistic** | Measuring agreement on categorical ratings | Kappa | Inter-rater reliability |
    """)

    # ------------------------
    st.header("📐 Estimation Module")

    with st.expander("❓ What is anticipated mean and standard deviation?"):
        st.markdown("These are expected values based on prior studies or pilot data. They are used to calculate the required sample size for estimating the population mean.\n\nUse published literature, small pilot studies, or historical data to estimate these values.")

    with st.expander("❓ What is anticipated proportion and when to use 0.5?"):
        st.markdown("Use your best guess of the proportion. If unsure, use 0.5 which gives the maximum required sample size and is considered conservative.\n\nThe closer your guess is to 0.5, the more uncertain the proportion — and the larger your required sample.")

    with st.expander("❓ What is ICC and why is it needed?"):
        st.markdown("The intra-class correlation coefficient (ICC) measures how similar subjects are within clusters. It's used in reliability studies and clustered designs.\n\nHigher ICC means more similarity within clusters and greater adjustment needed to sample size.")

    with st.expander("❓ What does k represent in ICC estimation?"):
        st.markdown("k is the number of raters, repeated measurements, or instruments used to measure each subject.\n\nIncreasing k reduces the required number of subjects.")

    # ------------------------
    st.header("🧪 Hypothesis Testing Module")

    with st.expander("❓ How is sample size calculated for comparing two means?"):
        st.markdown("It uses the expected means and standard deviation in each group, along with power and significance level, to compute the minimum required sample size per group.\n\nIf effect size is small, sample size will be large. If standard deviation is low, you need fewer subjects.")

    with st.expander("❓ What is Odds Ratio (OR) and how is it used in sample size calculation?"):
        st.markdown("OR measures association in case-control studies. Sample size depends on OR, proportion in controls, and power.\n\nLarger ORs (stronger effects) require fewer subjects to detect.")

    with st.expander("❓ What is Risk Ratio (RR)? How is it different from OR?"):
        st.markdown("RR is the ratio of risk between two groups, used in cohort studies. Unlike OR, RR is more intuitive but requires incidence data.\n\nUse RR when you follow people over time and can directly estimate risk.")

    with st.expander("❓ What is p₀ in case-control studies?"):
        st.markdown("p₀ is the proportion of exposure in the control group. It’s needed to estimate the expected number of exposed subjects.\n\nIf unknown, use estimates from literature or pilot studies.")

    # ------------------------
    st.header("📏 Reliability Module")

    with st.expander("❓ What is minimum acceptable reliability (ρ₀)?"):
        st.markdown("It’s the threshold ICC value you want to statistically demonstrate is exceeded. Used in hypothesis testing for reliability.\n\nFor example, showing ICC > 0.7 means your measurement is acceptably consistent.")

    with st.expander("❓ How does the number of raters (k) influence reliability estimates?"):
        st.markdown("More raters reduce the required sample size and increase precision of ICC.")  #or Kappa estimates.

    #with st.expander("❓ What is Kappa and when is it used?"):
    #    st.markdown("Kappa measures agreement between categorical ratings beyond chance. Useful in inter-rater reliability.\n\nValues above 0.7 generally indicate good agreement.")

    # ------------------------
    st.header("🧮 Input Field Explanations")

    with st.expander("❓ What is precision (d)?"):
        st.markdown("Precision (d) is the maximum allowed difference between sample estimate and true value. Smaller d = larger sample.\n\nFor example, if estimating blood pressure with ±5 units margin, d=5.")

    with st.expander("❓ What does alpha represent?"):
        st.markdown("Alpha is the significance level, usually 0.05. It reflects the chance of a Type I error (false positive).\n\nLower alpha (e.g., 0.01) gives stricter results but increases sample size.")

    with st.expander("❓ What does power represent?"):
        st.markdown("Power is the chance of detecting an effect if one exists. Usually set at 80% or 90%.\n\nHigher power = more reliable results, but needs a larger sample.")

    with st.expander("❓ What are Q₀ and Q₁ in group proportion?"):
        st.markdown("Q₀ is the proportion of total sample in the control group; Q₁ is the rest in the treatment group. They must sum to 1.\n\nFor equal group sizes, Q₀ = Q₁ = 0.5")

    with st.expander("❓ What is dropout rate?"):
        st.markdown("Dropout rate is the expected proportion of subjects who won’t complete the study. The sample size must be inflated to account for this.\n\nFor example, a 10% dropout means increasing your sample by 1.11x (i.e., n / (1 - 0.10))")

    # ------------------------
    st.markdown("---")
    st.success("I'll continue updating this page over time — feel free to share your thoughts or suggestions.")

    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")