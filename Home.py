import streamlit as st
import importlib
from PIL import Image
import base64
from io import BytesIO
import streamlit.components.v1 as components


st.set_page_config(page_title="Home | StudySizer", page_icon="🧮")
#st.sidebar.image("image.png", width=200)

# Convert the image to base64
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_image_base64("image.png")

st.sidebar.markdown(
    f"""
    <a href="https://studysizer.netlify.app/" target="_blank">
        <img src="data:image/png;base64,{image_base64}" width="200">
    </a>
    """,
    unsafe_allow_html=True
)
from modules.utils import inject_logo
#inject_logo()
#st.sidebar.title("StudySizer")

# Top-level dropdown
category = st.sidebar.selectbox("Select Category", ["-- Select --","About", "Estimation", "Comparison", "Diagnostic measures (Evaluation)", "Reliability", "Regression","FAQ"])

# 🌟 Show Home Page if nothing is selected
if category == "-- Select --":
    # Hide sidebar and header menu buttons
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

    # Welcome Message
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 64px;'>👋 Welcome to <span style="color: #00B2B2;">StudySizer</span></h1>
            <h3 style='font-size: 28px; color: gray;'>A Sample Size Calculator</h3>
            <p style='font-size: 20px; max-width: 700px; margin: auto;'>
                Empowering researchers, healthcare professionals, and students with statistically sound sample size calculations
                for various study designs including estimation, testing, regression, and reliability.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Get Started Button
    #if st.button("🚀 Get Started", use_container_width=True):
    #    st.switch_page("Home.py")

    #st.markdown("""
    #    <div style='text-align: center;'>
    #        <p style='font-size: 20px; max-width: 700px; margin: auto;'>
    #            To get started, please select a category from the dropdown list.
    #        </p>
    #    </div>
    #""", unsafe_allow_html=True)

    #st.markdown("<br>", unsafe_allow_html=True)
    
    st.image("PageButtonDiagram.jpeg")

elif category == "About":
    # Load logo image (your uploaded icon)
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
    " the required sample size for their studies. It offers a user-friendly interface and supports a range of statistical"
    " methods for different study designs. Here, you will also be able to find the required sample size calculation formulas, along with the corresponding references.")

    # Inject CSS and image at the top of the sidebar
    #col1,col2=st.columns(2)
    #with col1:
    #    st.image("Sample size Explained.png")
    #with col2:
    #    st.image("Sample size Explained_2.png")

    st.title("🔍 Choosing the Right Sample Size Method: A Practical Guide")

    st.markdown("Selecting the correct method for sample size estimation is crucial in ensuring your study is adequately powered and produces reliable, interpretable results. Below, we explain when to use each method, with real-world examples from healthcare, epidemiology, and clinical research.")

    st.markdown("""
        ## 📌 Proportion Estimation
        #### When to use:
        Use this when your main outcome is a proportion (percentage). It answers questions like “How many?” or “What fraction?”

        #### Real Example:
        A public health survey wants to estimate the prevalence of diabetes in an urban population. If prior data suggest ~12% prevalence, you would use proportion estimation to calculate how many participants you need to estimate this percentage with a desired precision (say, ±2%).
                
        ## 📌 Normal Mean Estimation

        #### When to use:
        Use this when your outcome is continuous and approximately normally distributed.

        #### Real Example:
        A nutritionist wants to estimate the average hemoglobin level in pregnant women attending a primary care clinic. Since hemoglobin levels typically follow a normal distribution, normal mean estimation is appropriate.

        ## 📌 Skewed Normal Mean Estimation

        #### When to use:
        When data are continuous but skewed (not symmetric). Instead of SD, this method uses the coefficient of variation (CV).

        #### Real Example:
        Hospital stay lengths often have a long right tail (a few patients stay much longer). To estimate the average length of stay in an orthopedic ward, a skewed-normal approach (using CV) is more reliable than assuming normality.

        ## 📌 Gamma Mean Estimation

        #### When to use:
        When outcomes follow a Gamma distribution, often used for positively skewed continuous data (e.g., costs, waiting times, lab measures that can’t be negative). Precision can be defined in absolute or relative terms.

        #### Real Example:
        A health economist wants to estimate the average cost of hospitalization for COVID-19 patients. Since costs are positive and right-skewed, the Gamma distribution provides a natural framework for sample size calculation.

        ## 📌 Gamma Mean Comparison

        #### When to use:
        When comparing the means of two groups that follow a Gamma distribution.

        #### Real Example:
        Researchers want to compare the average treatment cost for patients receiving Drug A versus Drug B for hypertension. As costs are skewed, Gamma mean comparison ensures sample size planning is tailored to the data’s distribution.

        ## 📌 Sensitivity & Specificity Estimation

        #### When to use:
        When validating a new diagnostic or screening tool.

        #### Real Example:
        A team develops a rapid antigen test for dengue. They need to estimate sensitivity (true positive rate) and specificity (true negative rate) with precision to confirm its reliability against PCR (the gold standard).

        ## 📌 Intraclass Correlation (ICC) Estimation

        #### When to use:
        For reliability and agreement studies where multiple raters or repeated measures are involved.

        #### Real Example:
        Three radiologists independently interpret the same set of X-rays to assess bone fractures. ICC tells how consistent their ratings are across patients.

        ## 📌 ICC Hypothesis Testing

        #### When to use:
        When testing if ICC meets a required threshold (e.g., acceptable reliability).

        #### Real Example:
        In a new telemedicine platform, you may want to test if ICC > 0.80 for blood pressure readings compared to manual sphygmomanometer readings.

        ## 📌 Chen’s Kappa Estimation

        #### When to use:
        When assessing agreement between two or more raters on categorical data.

        #### Real Example:
        Two pathologists independently classify biopsy slides as malignant or benign. Kappa measures the degree of agreement beyond chance.

        ## 📌 Two Sample Normal Mean Hypothesis

        #### When to use:
        When comparing average outcomes between two independent groups.

        #### Real Example:
        A randomized controlled trial (RCT) compares average systolic blood pressure between patients on a new antihypertensive vs. standard care.

        ## 📌 One-Way ANOVA

        #### When to use:
        When comparing means across three or more groups.

        #### Real Example:
        A clinical trial with three arms: Drug A, Drug B, and placebo, comparing mean reduction in HbA1c after 12 weeks.

        ## 📌 Paired t-Test

        #### When to use:
        For before-after studies with the same subjects.

        #### Real Example:
        Measuring cholesterol levels in patients before and after starting a statin. Since the same patients are measured twice, a paired test is used.

        ## 📌 Two Sample Proportion Hypothesis Testing

        #### When to use:
        When comparing two independent proportions.

        #### Real Example:
        Comparing smoking prevalence between rural and urban adolescents. The outcome (smoker/non-smoker) is binary.

        ## 📌 Paired Proportion (McNemar Test)

        #### When to use:
        For paired binary outcomes.

        #### Real Example:
        An awareness campaign is conducted in a factory. Workers’ smoking status (Yes/No) is measured before and after the campaign. McNemar’s test helps assess change.

        ## 📌 Poisson Rate Comparison

        #### When to use:
        When comparing event rates (counts per unit time/person-time).

        #### Real Example:
        Comparing infection rates per 1,000 catheter days between two hospital ICUs. Since data are event counts over exposure time, Poisson models apply.

        ## 📌 Correlation Test

        #### When to use:
        When assessing linear association between two continuous variables.

        #### Real Example:
        Checking whether BMI correlates with total cholesterol among adults in a health check-up camp.

        ## 📌 Case-Control Design

        #### When to use:
        When studying rare outcomes retrospectively.

        #### Real Example:
        A case-control study on lung cancer identifies patients (cases) and compares their smoking exposure with matched controls.

        ## 📌 Cohort Design

        #### When to use:
        For prospective studies comparing risks between groups.

        #### Real Example:
        A 5-year cohort study compares incidence of heart attack in diabetics vs. non-diabetics.

        ## 📌 Survival Analysis – Log-Rank Test

        #### When to use:
        For comparing survival curves across groups.

        #### Real Example:
        In an oncology trial, patients are randomized to chemotherapy vs. immunotherapy. The log-rank test compares time to progression.

        ## 📌 Linear Regression

        #### When to use:
        When predicting a continuous outcome from predictors.

        #### Real Example:
        A regression model predicts systolic BP from age, BMI, and salt intake.

        ## 📌 Logistic Regression

        #### When to use:
        When the outcome is binary.

        #### Real Example:
        A logistic regression model predicts the odds of gestational diabetes based on maternal age, BMI, and family history.
    """)
    st.markdown("---")
    st.markdown("**Developed by [Rajesh Majumder]**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")

# 🔁 If category is selected, proceed with nested dropdowns
else:
    method = None
    if category == "Estimation":
        method = st.sidebar.selectbox("Choose Method", [
            "Proportion Estimation", 
            "Normal Mean Estimation", 
            "Skewed Normal Mean Estimation",
            "Gamma Mean Estimation"
        ])
    elif category == "Comparison":
        method = st.sidebar.selectbox("Choose Test", [
            "Two Sample Normal Mean Comparison",
            "One way ANOVA",
            "Paired t test",
            "Two Sample Proportion Comparison",
            "Paired proportion Mc Nemar test",
            "Two Sample Poisson Rates Comparison",
            "Two Sample Gamma Mean Comparison",
            "Two Sample Correlation Comparison",
            "Case Control Design",
            "Cohort Design",
            "Survival Analysis Log rank Test for Two Groups"
        ])
    elif category == "Diagnostic measures (Evaluation)":
        method = st.sidebar.selectbox("Choose Test", [
            "Sensitivity and Specificity"
        ])
    elif category == "Reliability":
        method = st.sidebar.selectbox("Choose Method", [
            "Intraclass Correlation Estimation",
            "Two Sample Intraclass Correlation Comparison",
            "Cohen's Kappa Estimation"
        ])
    elif category == "Regression":
        method = st.sidebar.selectbox("Choose Model", [
            "Linear Regression",
            "Logistic Regression"
        ])
    elif category == "FAQ":
        from modules import FAQ
        FAQ.main()

    # 🔁 Module Loader
    if method:
        module_name = method.strip().replace(" ", "_").replace("-", "_").replace("__", "_")
        try:
            module = importlib.import_module(f"modules.{module_name}")
            module.main()
        except ModuleNotFoundError:
            st.error(f"⚠️ Module for `{module_name}` not found.")
        except AttributeError:
            st.error(f"⚠️ `{module_name}` is missing a `main()` function.")



####################
## Chat bot Entry ##
####################

def tawk_to_component(property_id: str, widget_id: str):
    """
    A Streamlit component to embed the Tawk.to chat widget.

    This function uses a workaround to inject the Tawk.to script into the main 
    document's head. This is necessary for the widget to float correctly over 
    the entire Streamlit app, rather than being confined to a specific component area.

    Args:
        property_id (str): Your Tawk.to Property ID.
        widget_id (str): Your Tawk.to Widget ID (e.g., 'default' or a custom ID).
    """
    # The JavaScript code that will be executed in the browser.
    # It creates a script element and adds it to the parent document's head.
    component_code = f"""
        <!-- This component is designed to be invisible (height=0) -->
        <!-- It injects the Tawk.to script into the main Streamlit app document. -->
        <script type="text/javascript">
            // This is the standard Tawk.to script
            var Tawk_API = Tawk_API || {{}}, Tawk_LoadStart = new Date();
            
            // We are creating a function that will be executed immediately
            (function() {{
                // Check if the Tawk.to script is already present in the parent document
                const tawkToSrc = 'https://embed.tawk.to/{property_id}/{widget_id}';
                if (parent.document.querySelector(`script[src="${{tawkToSrc}}"]`)) {{
                   // If the script is already there, don't add it again.
                    // This prevents duplication when Streamlit re-runs the script.
                    return;
                }}

                // Create a new script element for the Tawk.to widget
                var s1 = parent.document.createElement("script");
                s1.async = true;
                s1.src = tawkToSrc;
                s1.charset = 'UTF-8';
                s1.setAttribute('crossorigin', '*');
                
                // Find the first script tag in the parent document
                var s0 = parent.document.getElementsByTagName("script")[0];
                
                // Insert the new script element before the first script tag
                // This is the recommended way to add async scripts.
                if (s0) {{
                    s0.parentNode.insertBefore(s1, s0);
                }} else {{
                    // If no script tags are found, just append it to the head.
                    parent.document.head.appendChild(s1);
                }}
            }})();
        </script>
    """
    
    # Use st.components.v1.html to render the JS code
    # Setting height=0 makes the component invisible.
    components.html(component_code, height=0)

default_prop_id = st.secrets["auth_key"]
default_widget_id = st.secrets["bot_key"]

tawk_to_component(property_id=default_prop_id, widget_id=default_widget_id)
