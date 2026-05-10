import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# =========================================================
# PAGE CONFIG
# =========================================================

def main():
    #st.set_page_config(page_title="StudySizer | Simulation Sample Size",page_icon="🧮",layout="centered")

    st.title("Simulation-Based Sample Size Calculator")

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
    This module estimates sample size using:

    - Logistic regression (Odds Ratio)
    - Modified Poisson regression (Relative Risk)
    - Monte Carlo simulation
    """)

    # =========================================================
    # REMOVE NUMBER INPUT SPIN BUTTONS
    # =========================================================

    st.markdown(
        """
        <style>
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        input[type=number] {
            -moz-appearance: textfield;
        }

        button[data-testid="stBaseButton-header"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # =========================================================
    # HISTORY STORAGE
    # =========================================================

    if "simulation_history" not in st.session_state:
        st.session_state.simulation_history = []

    # =========================================================
    # SIDEBAR
    # =========================================================

    st.sidebar.header("🔧 Input Parameters")

    # ---------------------------------------------------------
    # EFFECT TYPE

    effect_type = st.sidebar.radio(
        "Effect Type",
        ["OR", "RR"]
    )

    # ---------------------------------------------------------
    # EFFECT SIZE

    effect = st.sidebar.number_input(
        "Effect Size (OR/RR)",
        value=1.2,
        min_value=0.01,
        step=0.1,
        format="%.6g",
        help="Enter Odds Ratio or Relative Risk."
    )

    effect_unit = st.sidebar.number_input(
        "Effect Per Unit Increase",
        value=10.0,
        min_value=0.01,
        format="%.6g",
        help="""
    Specify the unit increase in exposure for which the effect size is defined.

    Examples:

    • OR = 1.20 per 10 µg/m³ PM2.5 increase
    → Effect Size = 1.20
    → Effect Unit = 10

    • RR = 1.05 per 1 year increase in age
    → Effect Unit = 1

    • OR = 1.30 per 5 kg/m² BMI increase
    → Effect Unit = 5

    Internally:

    β = log(Effect Size) / Effect Unit
    """
    )

    # ---------------------------------------------------------
    # EXPOSURE DISTRIBUTION

    exposure_dist = st.sidebar.selectbox(
        "Exposure Distribution",
        ["normal", "lognormal", "gamma", "median_iqr"]
    )

    # ---------------------------------------------------------
    # DISTRIBUTION PARAMETERS

    if exposure_dist in ["normal", "lognormal", "gamma"]:

        mean_x = st.sidebar.number_input(
            "Exposure Mean",
            value=50.0,
            format="%.6g"
        )

        sd_x = st.sidebar.number_input(
            "Exposure SD",
            value=15.0,
            format="%.6g"
        )

        median_x = None
        q1 = None
        q3 = None

    else:

        median_x = st.sidebar.number_input(
            "Median",
            value=40.0,
            format="%.6g"
        )

        q1 = st.sidebar.number_input(
            "Q1",
            value=30.0,
            format="%.6g"
        )

        q3 = st.sidebar.number_input(
            "Q3",
            value=60.0,
            format="%.6g"
        )

        mean_x = None
        sd_x = None

    # ---------------------------------------------------------
    # TRANSFORMATION

    log_transform = st.sidebar.checkbox(
        "Log Transform Exposure"
    )

    # ---------------------------------------------------------
    # BASELINE RISK

    p0 = st.sidebar.number_input(
        "Baseline Probability",
        value=0.20,
        min_value=0.0001,
        max_value=0.9999,
        format="%.6g"
    )

    # ---------------------------------------------------------
    # SAMPLE SIZE RANGE

    st.sidebar.markdown("---")

    n_min = st.sidebar.number_input(
        "Minimum Sample Size",
        value=100
    )

    n_max = st.sidebar.number_input(
        "Maximum Sample Size",
        value=2000
    )

    step = st.sidebar.number_input(
        "Step Size",
        value=100
    )

    # ---------------------------------------------------------
    # SIMULATION SETTINGS

    n_sim = st.sidebar.number_input(
        "Number of Simulations",
        value=300
    )

    target_power = st.sidebar.number_input(
        "Target Power (%)",
        value=80.0,
        format="%.6g"
    )/100

    alpha = st.sidebar.number_input(
        "Alpha",
        value=0.05,
        format="%.6g"
    )

    # ---------------------------------------------------------
    # DESIGN SETTINGS

    st.sidebar.markdown("---")

    design_effect = st.sidebar.number_input(
        "Design Effect",
        value=1.0,
        min_value=1.0,
        format="%.6g"
    )

    dropout = st.sidebar.number_input(
        "Dropout (%)",
        value=0.0,
        min_value=0.0,
        max_value=99.0,
        format="%.6g"
    ) / 100

    # =========================================================
    # HISTORY LABEL
    # =========================================================

    def make_history_label(entry):

        return (
            f"{entry['effect_type']} | "
            f"Effect={entry['effect']} per {entry['effect_unit']} unit | "
            f"Dist={entry['exposure_dist']} | "
            f"p0={entry['p0']} | "
            f"Power={entry['target_power']}"
        )

    # =========================================================
    # HISTORY SELECTOR
    # =========================================================

    selected_history = None
    hist_submit = False

    if len(st.session_state.simulation_history) > 0:

        st.subheader("📜 Previous Simulations")

        history_options = [
            make_history_label(x)
            for x in st.session_state.simulation_history
        ]

        selected_label = st.selectbox(
            "Select Previous Simulation",
            history_options
        )

        if selected_label:

            selected_history = next(
                (
                    item
                    for item in st.session_state.simulation_history
                    if make_history_label(item)
                    == selected_label
                ),
                None
            )

            hist_submit = st.button(
                "🔁 Recalculate Selected Simulation"
            )

    # =========================================================
    # SIMULATION FUNCTION
    # =========================================================

    def simulate_sample_size():

        beta = np.log(effect) / effect_unit

        # -----------------------------------------------------
        # GENERATE EXPOSURE

        def gen_X(n):

            if exposure_dist == "normal":

                X = np.random.normal(
                    mean_x,
                    sd_x,
                    n
                )

            elif exposure_dist == "lognormal":

                meanlog = np.log(
                    mean_x**2 /
                    np.sqrt(sd_x**2 + mean_x**2)
                )

                sdlog = np.sqrt(
                    np.log(
                        1 + (sd_x**2 / mean_x**2)
                    )
                )

                X = np.random.lognormal(
                    meanlog,
                    sdlog,
                    n
                )

            elif exposure_dist == "gamma":

                shape = (mean_x / sd_x)**2

                scale = (sd_x**2) / mean_x

                X = np.random.gamma(
                    shape,
                    scale,
                    n
                )

            else:

                sd_est = (q3 - q1) / 1.35

                X = np.random.normal(
                    median_x,
                    sd_est,
                    n
                )

            if log_transform:

                X = np.log(
                    X - np.min(X) + 1
                )

            return X

        # -----------------------------------------------------
        # SAFE ROOT FINDER

        def safe_root(f):

            grid = np.arange(
                -1000,
                1000,
                5
            )

            vals = np.array(
                [f(g) for g in grid]
            )

            idx = np.where(
                np.diff(np.sign(vals)) != 0
            )[0]

            if len(idx) == 0:

                raise ValueError(
                    "Intercept calibration failed."
                )

            lo = grid[idx[0]]
            hi = grid[idx[0] + 1]

            return brentq(f, lo, hi)

        # -----------------------------------------------------
        # CALIBRATE INTERCEPT

        Xtmp = gen_X(10000)

        Xtmp_c = Xtmp - np.mean(Xtmp)

        if effect_type == "OR":

            def f(a):

                p = 1 / (
                    1 + np.exp(
                        -(a + beta * Xtmp_c)
                    )
                )

                return np.mean(p) - p0

        else:

            def f(a):

                p = np.exp(
                    a + beta * Xtmp_c
                )

                p[p > 0.999] = 0.999

                return np.mean(p) - p0

        alpha0 = safe_root(f)

        # -----------------------------------------------------
        # SAMPLE SIZE LOOP

        sample_sizes = np.arange(
            n_min,
            n_max + step,
            step
        )

        powers = []

        progress = st.progress(0)

        for i, n in enumerate(sample_sizes):

            sig = []

            for s in range(n_sim):

                X = gen_X(n)

                Xc = X - np.mean(X)

                # -------------------------------------------------
                # LOGISTIC MODEL

                if effect_type == "OR":

                    p = 1 / (
                        1 + np.exp(
                            -(alpha0 + beta * Xc)
                        )
                    )

                    Y = np.random.binomial(
                        1,
                        p
                    )

                    X_design = sm.add_constant(Xc)

                    model = sm.Logit(
                        Y,
                        X_design
                    )

                # -------------------------------------------------
                # MODIFIED POISSON

                else:

                    p = np.exp(
                        alpha0 + beta * Xc
                    )

                    p[p > 0.999] = 0.999
                    p[p < 0.0001] = 0.0001

                    Y = np.random.binomial(
                        1,
                        p
                    )

                    X_design = sm.add_constant(Xc)

                    model = sm.GLM(
                        Y,
                        X_design,
                        family=sm.families.Poisson(
                            link=sm.families.links.log()
                        )
                    )

                try:

                    fit = model.fit(disp=0)

                    pval = fit.pvalues[1]

                except:

                    pval = 1

                sig.append(
                    pval < alpha
                )

            power_n = np.mean(sig)

            powers.append(power_n)

            progress.progress(
                (i + 1) / len(sample_sizes)
            )

        # -----------------------------------------------------
        # RESULTS

        results = pd.DataFrame({
            "Sample Size": sample_sizes,
            "Power": powers
        })

        valid = results.loc[
            results["Power"] >= target_power,
            "Sample Size"
        ]

        if len(valid) == 0:

            base_n = np.nan
            adjusted_n = np.nan

        else:

            base_n = valid.min()

            adjusted_n = np.ceil(
                base_n *
                design_effect /
                (1 - dropout)
            )

        x_example = gen_X(10000)

        return results, base_n, adjusted_n, x_example

    # =========================================================
    # BUTTONS
    # =========================================================

    go = st.button("Calculate Sample Size")

    # =========================================================
    # MAIN EXECUTION
    # =========================================================

    if go or hist_submit:

        # -----------------------------------------------------
        # RECALCULATE FROM HISTORY

        if hist_submit and selected_history:

            effect_type = selected_history["effect_type"]
            effect = selected_history["effect"]
            effect_unit = selected_history["effect_unit"]

            exposure_dist = selected_history["exposure_dist"]

            mean_x = selected_history["mean_x"]
            sd_x = selected_history["sd_x"]

            median_x = selected_history["median_x"]
            q1 = selected_history["q1"]
            q3 = selected_history["q3"]

            log_transform = selected_history["log_transform"]

            p0 = selected_history["p0"]

            n_min = selected_history["n_min"]
            n_max = selected_history["n_max"]
            step = selected_history["step"]

            n_sim = selected_history["n_sim"]

            target_power = selected_history["target_power"]

            alpha = selected_history["alpha"]

            design_effect = selected_history["design_effect"]

            dropout = selected_history["dropout"]

        # -----------------------------------------------------
        # SAVE CURRENT INPUTS

        else:

            new_history = {

                "effect_type": effect_type,
                "effect": effect,
                "effect_unit": effect_unit,

                "exposure_dist": exposure_dist,

                "mean_x": mean_x,
                "sd_x": sd_x,

                "median_x": median_x,
                "q1": q1,
                "q3": q3,

                "log_transform": log_transform,

                "p0": p0,

                "n_min": n_min,
                "n_max": n_max,
                "step": step,

                "n_sim": n_sim,

                "target_power": target_power,

                "alpha": alpha,

                "design_effect": design_effect,

                "dropout": dropout
            }

            st.session_state.simulation_history.append(
                new_history
            )

        # -----------------------------------------------------
        # RUN

        with st.spinner("Running simulations..."):

            results, base_n, adjusted_n, x_example = \
                simulate_sample_size()

        # =====================================================
        # OUTPUT
        # =====================================================

        st.success("Simulation Completed")

        col1, col2 = st.columns(2)

        col1.metric(
            "Base Sample Size",
            value=base_n
        )

        col2.metric(
            "Adjusted Sample Size",
            value=adjusted_n
        )

        # -----------------------------------------------------
        # TABLE

        st.subheader("Power Table")

        st.dataframe(results,use_container_width=True)

        # -----------------------------------------------------
        # POWER CURVE

        st.subheader("Power Curve")

        fig, ax = plt.subplots()

        ax.plot(
            results["Sample Size"],
            results["Power"],
            marker="o"
        )

        ax.axhline(
            target_power,
            linestyle="--"
        )

        ax.set_xlabel(
            "Sample Size"
        )

        ax.set_ylabel(
            "Power"
        )

        ax.set_title(
            "Simulation-Based Power Curve"
        )

        ax.grid(True)

        st.pyplot(fig)

        # -----------------------------------------------------
        # EXPOSURE DISTRIBUTION

        st.subheader(
            "Distribution of Exposure Variable (X)"
        )

        col1, col2 = st.columns(2)

        # Histogram

        with col1:

            fig1, ax1 = plt.subplots()

            ax1.hist(
                x_example,
                bins=30
            )

            ax1.set_title(
                "Histogram of Exposure"
            )

            ax1.set_xlabel(
                "Exposure"
            )

            ax1.set_ylabel(
                "Frequency"
            )

            st.pyplot(fig1)

        # Boxplot

        with col2:

            fig2, ax2 = plt.subplots()

            ax2.boxplot(x_example)

            ax2.set_title(
                "Boxplot of Exposure"
            )

            ax2.set_ylabel(
                "Exposure"
            )

            st.pyplot(fig2)

        # -----------------------------------------------------
        # INTERPRETATION

        st.markdown("---")

        if np.isnan(adjusted_n):

            st.warning("""
            Target power was not achieved
            within the selected sample size range.
            """)

        else:

            st.markdown(f"""
            Required adjusted sample size:

            # {int(adjusted_n)}
            """)

    # =========================================================
    # METHODOLOGY
    # =========================================================

    st.markdown("---")

    with st.expander("Show Methodology"):

        st.markdown("""
        ## Simulation-Based Sample Size Calculation

        Steps:

        1. Generate exposure variable
        2. Generate binary outcome
        3. Fit regression model
        4. Test significance
        5. Repeat simulations
        6. Estimate empirical power
        7. Find minimum sample size

        ### Supported Models

        - Logistic regression (OR)
        - Modified Poisson regression (RR)

        ### Advantages

        - Handles skewed exposure
        - Handles nonlinear distributions
        - More realistic than closed-form methods
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