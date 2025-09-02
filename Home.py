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
    col1,col2=st.columns(2)
    with col1:
        st.image("Sample size Explained.png")
    with col2:
        st.image("Sample size Explained_2.png")

    st.title("🔍 When to Use Which Sample Size Calculator — A Practical Guide")

    st.markdown("Let’s break down **when and why** you’d use each of the sample size methods available in the app.")

    st.markdown("""
        ### Proportion Estimation:
        Use this when you're trying to estimate a single proportion—like what percentage of people in a city have diabetes or how many support a health policy. It’s perfect for cross-sectional surveys or baseline assessments where your goal is just to understand “how much” or “how many.”
        
        ### Normal Mean Estimation:
        Go for this when you're estimating an average—say, the average systolic blood pressure in a population. This works best if your outcome is normally distributed. You’ll commonly use this in clinical studies, nutrition research, or any setting where averages matter.
        
        ### Skewed Normal Mean Estimation:
        Not everything follows a normal curve. Think of healthcare costs or lengths of hospital stay—these are usually skewed. If your variable is skewed but you still want to estimate a mean, use this method. It relies on the coefficient of variation (CV) instead of the standard deviation.
        
        ### Sensitivity & Specificity:
        This is for those working on diagnostic accuracy. If you’ve built a new test or screening tool and want to validate its performance, this calculator helps you plan for estimating sensitivity and specificity with precision.
        
        ### Intraclass Correlation (ICC) Estimation:
        Use this in reliability studies. For example, if multiple raters are evaluating X-rays or you're checking how consistent measurements are within clusters, ICC tells you how much of the variability is due to real differences vs. measurement noise.
        
        ### ICC Hypothesis Testing:
        Similar to ICC estimation, but here you're testing a hypothesis—maybe you want to prove that your new tool has better reliability than a certain threshold (e.g., ICC > 0.70). Great for multi-site reliability checks or instrument validation.
        
        ### Chen’s Kappa Estimation:
        Need to measure agreement between raters for categorical data? Kappa is your go-to. Use this when you're dealing with yes/no, present/absent kind of decisions—like whether two doctors agree on a diagnosis. Works with 2 or more raters.
        
        ### Two Sample Normal Mean Hypothesis:
        You’ll use this in classic setups like comparing treatment vs. control groups. Say you want to know if a new diet reduces blood sugar more than the standard one. If your outcome is continuous and your groups are independent, this is your choice.
        
        ### One Way ANOVA:
        Got three or more groups to compare? That’s where ANOVA comes in. Use this in multi-arm clinical trials, educational experiments, or policy evaluations where multiple interventions are being compared.
        
        ### Paired t-Test:
        Perfect for before-after studies. If you’re giving the same group a treatment and measuring the difference in some continuous outcome, this is your test. Think pre- and post-intervention blood pressure, or step count after using a fitness app.
        
        ### Two Sample Proportion Hypothesis Testing:
        Here, you’re comparing proportions—for example, the percentage of smokers in two different cities or success rates of two drugs. If your outcome is binary and your groups are independent, this is the tool.
        
        ### Paired Proportion (McNemar Test):
        This is for comparing paired binary outcomes—think of it like a paired t-test but for yes/no outcomes. Example: Did smoking status change before and after an awareness campaign within the same group?
        
        ### Two Sample Poisson Rate Comparison:
        If you’re comparing event rates (like infections per 1000 patient-days), and you expect count data over time or person-time, use this. Common in hospital-based studies or epidemiological surveillance.
        
        ### Correlation Test:
        Trying to understand if two continuous variables move together? Correlation is what you want. Like checking the relationship between BMI and cholesterol. This helps you figure out if a linear relationship exists and how strong it is.
        
        ### Case Control Design: 
        This is your tool for retrospective studies. You’ve got cases and controls, and you're looking backward to see if exposure rates differ. It's especially useful in rare disease research, where measuring odds is more practical than risk.
        
        ### Cohort Design:
        Ideal for longitudinal studies. You follow two groups over time and compare risk or incidence rates. Use this when you're tracking outcomes like heart attacks, recovery, or vaccine breakthrough infections.
        
        ### Survival Analysis – Log-Rank Test:
        When your outcome is time-to-event, this test is key. Say you're comparing how long patients survive under two treatments. Use this in oncology studies, clinical trials, or anywhere time plays a critical role.
        
        ### Linear Regression:
        This one’s for predictive modeling with continuous outcomes. Want to see how education, income, and age affect blood pressure? Use linear regression—it tells you how much change you can expect in the outcome for each predictor.
        
        ### Logistic Regression:
        When your outcome is binary—like disease or no disease—logistic regression helps you understand how different predictors influence the odds. Common in case-control and risk modeling studies.
        
        ### About:
        Think of this as your starting point. It gives you a quick tour of the app, how things are organized, and what each section is about.
        
        ### FAQ:
        Got doubts? Confused about what a dropdown means or why you’re being asked for alpha and beta? The FAQ section explains these, in plain language.
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
            "Skewed Normal Mean Estimation"
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
