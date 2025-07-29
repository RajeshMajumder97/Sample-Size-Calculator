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
category = st.sidebar.selectbox("Select Category", ["-- Select --","About", "Estimation", "Testing", "Diagnostic measures (Evaluation)", "Reliability", "Regression","FAQ"])

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

    st.markdown("""
        <div style='text-align: center;'>
            <p style='font-size: 20px; max-width: 700px; margin: auto;'>
                To get started, please select a category from the dropdown list.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
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
    " methods for different study designs. The advantage of this tool is, it also gives the required sample size calculation formulas along with the references.")

    st.markdown("Hi, I am Rajesh, a Ph.D. student in Biostatistics. If you find this tool useful, please cite it as:")
    st.markdown("*StudySizer: A Sample Size Calculator, developed by Rajesh Majumder ([https://studysizer.netlify.app/](https://studysizer.netlify.app/))*")

    st.markdown("**If you want to reach me :**")
    st.markdown("**Email:** rajeshnbp9051@gmail.com")
    st.markdown("**Website:** [https://rajeshmajumderblog.netlify.app/](https://rajeshmajumderblog.netlify.app/)")


    # Inject CSS and image at the top of the sidebar

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

    render_section("Two Samples Poisson Rate Comparison",
                "Comparing the event rates (e.g., counts per person-time) between two independent groups.",
                "Rate Ratio (RR) or Difference in Event Rates",
                "Incidence rate comparison in cohort studies, adverse event rate comparison, infection rates, hospital visit rates, etc.")

    render_section("Two Sample Proportion Hypothesis Testing",
                "Comparing the proportion of events between two independent groups.",
                "Proportion difference",
                "Treatment effectiveness, intervention studies.")

# 🔁 If category is selected, proceed with nested dropdowns
else:
    method = None
    if category == "Estimation":
        method = st.sidebar.selectbox("Choose Method", [
            "Proportion Estimation", 
            "Normal Mean Estimation", 
            "Skewed Normal Mean Estimation"
        ])
    elif category == "Testing":
        method = st.sidebar.selectbox("Choose Test", [
            "Two Sample Normal Mean Hypothesis Testing",
            "One way ANOVA",
            "Paired t test",
            "Two Sample Proportion Hypothesis Testing",
            "Paired proportion Mc Nemar test",
            "Two Sample Poisson Rates Comparison",
            "Correlation Test",
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
            "Intraclass Correlation Hypothesis Testing",
            "Cohen's Kappa Estimation"
        ])
    elif category == "Regression":
        method = st.sidebar.selectbox("Choose Model", [
            "Linear Regression",
            "Logistic Regression",
            "Test Module"
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
