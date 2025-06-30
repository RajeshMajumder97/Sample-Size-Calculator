import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
from scipy.special import erf

st.set_page_config(page_title="One way ANOVA",
                   page_icon="🧊")

st.markdown("""
    <style>
    button[data-testid="stBaseButton-header"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.title("Sample Size Calculation for One way ANOVA")

## Functuion
def calculate_anova_sample_size(effect_size, alpha, power, k,dpt):
    """
    Calculate the required sample size per group for one-way ANOVA using the noncentral F-distribution.
    
    Parameters:
    effect_size (float): Cohen's f effect size
    alpha (float): Significance level
    power (float): Desired power
    k (int): Number of groups
    
    Returns:
    tuple: (sample size per group, total sample size)
    """
    # Get Z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - (1-alpha) / 2)  # Two-tailed test
    z_beta = stats.norm.ppf(power)
    
    # Start with an initial guess for sample size
    n = 2
    
    while True:
        # Degrees of freedom
        df1 = k - 1  # Between-group df
        df2 = k * (n - 1)  # Within-group df
        
        # Compute the critical F-value
        f_crit = stats.f.ppf(1 - (1-alpha), df1, df2)
        
        # Compute the noncentrality parameter
        lambda_ncp = effect_size * np.sqrt(k * n)
        
        # Compute power using the noncentral F-distribution
        power_calc = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_ncp**2)
        
        # Check if computed power meets or exceeds desired power
        if power_calc >= power:
            break
        
        # Increase sample size and try again
        n += 1
        n= n/(1-dpt)

    # Return per-group and total sample size
    return lambda_ncp**2,f_crit,df1,df2,power_calc,n, k * n,k

# Initialize history store
if "anova_history" not in st.session_state:
    st.session_state.anova_history = []




esize = st.sidebar.number_input("Effect size (Cohen's f)",value=0.25,min_value=0.0,max_value=1.0)
st.sidebar.text("0.10 = Small effect size\n 0.25= Medium effect size\n 0.40= Large effect size")
power= st.sidebar.number_input("Power (%)", value=80.0,min_value=0.0,max_value=100.0)
KK=st.sidebar.number_input("Number of groups (k)",value=5,min_value=3)
drpt= st.sidebar.number_input("Drop-Out (%)",min_value=0.0,value=0.0,max_value=100.0)
go= st.button("Calculate Sample Size")


# Helper to generate label for dropdown
def make_anova_history_label(esize, KK, power, drpt):
        return f"Cohen's f={esize}, k={KK}, Power={power}%, DropOut={drpt}%"

# Select from history
selected_history = None
selected_label = None

if st.session_state.anova_history:
    st.subheader("📜 Select from Past Inputs (Click & Recalculate)")
    anova_options = [make_anova_history_label(**entry) for entry in st.session_state.anova_history]
    selected_label = st.selectbox("Choose a past input set:", anova_options, key="anova_history_selector")

    if selected_label:
        selected_history = next((item for item in st.session_state.anova_history
                                 if make_anova_history_label(**item) == selected_label), None)
        hist_submit = st.button("🔁 Recalculate from Selected History")
    else:
        hist_submit = False
else:
    hist_submit = False


if go or hist_submit:

    if hist_submit and selected_history:
        # Use selected history
        esize= selected_history["esize"]
        KK= selected_history["KK"]
        power = selected_history["power"]
        drpt = selected_history["drpt"]
    else:
        # Add current input to history
        new_entry = {
            "esize":esize,
            "KK":KK,
            "power":power,
            "drpt":drpt
        }
        st.session_state.anova_history.append(new_entry)

    confidenceIntervals= [0.8,0.9,0.97,0.99,0.999,0.9999]
    out=[]

    for conf in confidenceIntervals:
        sample_size= calculate_anova_sample_size(effect_size=esize, alpha=conf, power=(power/100), k=KK,dpt=(drpt/100))
        out.append(sample_size)

    df= pd.DataFrame({
        "Confidence Levels (%)": [cl *100 for cl in confidenceIntervals],
        "No. of groups": [row[7] for row in out],
        "Noncentrality parameter": [row[0] for row in out],
        "F-critical value": [row[1] for row in out],
        "Upper df.": [row[2] for row in out],
        "Lower df.": [row[3] for row in out],
        "Calculated Power": [row[4] for row in out],
        "Sample sise par group wise": [row[5] for row in out],
        "Total Sample sise": [row[6] for row in out]
    })

    dds= calculate_anova_sample_size(effect_size=esize, alpha=0.95, power=(power/100), k=KK,dpt=(drpt/100))
    
    st.write(f"The study would need a total sample size of:")
    st.markdown(f"""
    <div style="display: flex; justify-content: center;">
        <div style="
            font-size: 36px;
            font-weight: bold;
            background-color: #48D1CC;
            padding: 10px;
            border-radius: 10px;
            text-align: center;">
            {int(dds[6])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write(f"""participants(i.e. <span style="background-color: #48D1CC; font-weight: bold; font-size: 26px;">{int(dds[5])}</span> participants at each group) to achive a power of {(power)}% and <span style="font-weight: bold; font-size: 26px;">95%</span> confidence level, for comparing {KK} different group means, where the non-centrality parameter is {round(dds[0],4)}, F-test critical value is {round(dds[1],4)} with numerator and denominator dfs are {int(dds[2])} and {int(dds[3])} respectively  where drop-out sample percentage is **{(drpt)}%**.""",unsafe_allow_html=True)
    st.subheader("List of Sample Sizes at other Confidence Levels")
    st.dataframe(df)


st.markdown("---")  # Adds a horizontal line for separation

st.subheader("📌 Formula for Sample Size Calculation")

st.markdown("### **One-way ANOVA Test Sample Size Formula**")

st.markdown("The sample size for One-way ANOVA is calculated using noncentral F-distribution:")
st.latex(r"""
\text{Power} = 1 - F_{\text{crit}, df_1, df_2}^{-1} (\alpha, \lambda)
""")

st.subheader("📌 Description of Parameters")

st.latex(r"""
F_{\text{crit}, df_1, df_2}^{-1}=\; \text{is the inverse cumulative distribution function (CDF) of the central F-distribution}
""")
st.latex(r"""
df_1 = k - 1=\; \text{degrees of freedom between groups}
""")
st.latex(r"""
df_2 = k(n - 1)=\; \text{degrees of freedom within groups}
""")
st.latex(r"""
\lambda = f \cdot k \cdot n=\; \text{is the noncentrality parameter}
""")
st.latex(r"""
n=\; \text{is the per-group sample size.}
""")
st.latex(r"""
k=\; \text{is the number of groups.}
""")
st.latex(r"""
f=\sqrt{\frac{\eta^2}{1-\eta^2}}=\; \text{is the Cohen's f : Effect size}
""")
st.latex(r"""
\eta=\frac{SS_{\text{Treatment}}}{SS_{\text{Total}}}
""")

st.subheader("📌 References")

st.markdown("""
1. **Cohen, J.** A power primer. Psychological bulletin vol. 112,1 (1992): 155-9. doi:10.1037//0033-2909.112.1.155
2. **Jan, Show-Li, and Gwowen Shieh.** Sample size determinations for Welch's test in one-way heteroscedastic ANOVA. The British journal of mathematical and statistical psychology vol. 67,1 (2014): 72-93. doi:10.1111/bmsp.12006
3. **Bujang, Mohamad Adam.** A Step-by-Step Process on Sample Size Determination for Medical Research. The Malaysian journal of medical sciences : MJMS vol. 28,2 (2021): 15-27. doi:10.21315/mjms2021.28.2.2
""")

st.markdown(""""
<span style="font-weight: bold; font-size: 26px;">Note that</span>, when your terget is the multiple comparisons use  **Bonferroni correction**:           
""",unsafe_allow_html=True)
st.latex(r"""
\alpha_{\text{adjusted}}=\frac{\alpha}{\text{Number of Comparisons}}
""")
st.markdown("""to adjust the significance level. This adjustment helps to control family-wise errow rate (FWER). Others are **Sidak Correction**,**Holm-Bonferroni**,**Benjamini-Hochberg**.""")


st.markdown("---")
st.subheader("Citation")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")


st.markdown("---")
st.markdown("**Developed by [Rajesh Majumder]**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")