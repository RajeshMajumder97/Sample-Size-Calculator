import streamlit as st
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(page_title="StydySizer | Home Page", page_icon="🧮")
# Load logo image (your uploaded icon)
logo = Image.open("image.png")

# Helper function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

logo_base64 = image_to_base64(logo)


# Centered title and subtitle
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 68px;'>StudySizer</h1>
        <h3 style='font-size: 38px;'>A Sample Size Calculator</h3>
    </div>
    <style>
    button[data-testid="stBaseButton-header"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.write("This open-source and free web application allows researchers, students, and professionals to calculate"
"the required sample size for their studies. It offers a user-friendly interface and supports a range of statistical"
" methods for different study designs. The advantage of this tool is, it also gives the required sample sie calculation formulas along with the references.")

st.markdown("Hi, I am Rajesh, a Ph.D. student in Biostatistics. If you find this tool useful, please cite it as:")
st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

st.markdown("**If you want to reach me :**")
st.markdown("**Email:** rajeshnbp9051@gmail.com")
st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")


# Inject CSS and image at the top of the sidebar
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 80px;
        background-image: url("data:image/png;base64,{logo_base64}");
        background-repeat: no-repeat;
        background-position: 20px 20px;
        background-size: 50px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.image("Sample size Explained.png")
st.image("Sample size Explained_2.png")


st.title("📘 Sample Size Calculation Scenarios")

st.write("A concise overview of when to apply each sample size calculation method based on study type and research objective.")

def render_section(title, use_when, measure, application, assumption=None):
    with st.container():
        st.subheader(title)
        st.markdown(f"**Use when:** {use_when}")
        if assumption:
            st.markdown(f"**Assumption:** {assumption}")
        st.markdown(f"**Measure:** {measure}")
        st.markdown(f"**Application:** {application}")

render_section("Case Control Design",
               "You're planning a retrospective study comparing exposure history between cases (with outcome) and controls (without outcome).",
               "Odds Ratio",
               "Epidemiology, identifying risk factors.")

render_section("Cohort Design",
               "You are conducting a longitudinal/prospective study to compare outcomes in exposed and non-exposed groups.",
               "Relative Risk (RR)",
               "Risk estimation, public health impact assessment.")

render_section("Correlation Test",
               "Studying the strength and direction of relationship between two continuous variables.",
               "Pearson correlation coefficient (r)",
               "Observational data analysis.")

render_section("Intraclass Correlation",
               "Measuring consistency/agreement within groups, raters, or clusters.",
               "ICC (Intraclass Correlation Coefficient)",
               "Reliability studies, cluster sampling.")

render_section("Linear Regression",
               "Analyzing how one or more predictors influence a continuous outcome.",
               "Regression coefficient",
               "Predictive modeling, public health determinants analysis.")

render_section("Logistic Regression",
               "The outcome is binary (e.g., disease/no disease) and you're modeling odds of occurrence.",
               "Odds Ratio",
               "Risk modeling, case-control designs.")

render_section("Normal Mean Estimation",
               "Estimating a population mean with a specific precision.",
               "—",
               "Nutritional studies, average clinical values.",
               assumption="Normal distribution")

render_section("One Way ANOVA",
               "Comparing means across 3 or more independent groups.",
               "F-statistic",
               "Experimental designs, treatment comparisons.")

render_section("Paired Proportion (McNemar Test)",
               "Comparing binary outcomes pre- and post-intervention for the same subjects.",
               "Proportion difference (paired)",
               "Health behavior change, intervention evaluation.")

render_section("Paired t-Test",
               "Measuring the change in a continuous variable before and after treatment in the same individuals.",
               "Mean difference (paired)",
               "Clinical trials, impact studies.")

render_section("Proportion Estimation",
               "Estimating a proportion or percentage (e.g., disease prevalence).",
               "Proportion",
               "Surveys, cross-sectional studies.")

render_section("Sensitivity and Specificity",
               "Evaluating diagnostic test performance.",
               "Sensitivity, Specificity",
               "Screening test validation.")

render_section("Skewed Normal Mean Estimation",
               "Estimating mean when the variable distribution is skewed (e.g., cost data).",
               "—",
               "Health economics, hospital resource planning.")

render_section("Survival Analysis – Log-rank Test (Two Groups)",
               "Comparing time-to-event data between two groups.",
               "Hazard Ratio (HR)",
               "Survival time, treatment comparison.")

render_section("Two Sample Normal Mean Hypothesis Testing",
               "Comparing the mean values of two independent groups.",
               "Mean difference",
               "RCTs, group comparison.")

render_section("Two Sample Proportion Hypothesis Testing",
               "Comparing the proportion of events between two independent groups.",
               "Proportion difference",
               "Treatment effectiveness, intervention studies.")
