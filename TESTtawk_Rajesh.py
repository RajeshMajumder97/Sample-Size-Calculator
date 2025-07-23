import streamlit as st
import streamlit.components.v1 as components

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


# --- Streamlit App ---

# Set page configuration
st.set_page_config(page_title="Tawk.to Demo", layout="wide")

# Sidebar for configuration
st.sidebar.title("⚙️ Tawk.to Configuration")
st.sidebar.markdown(
    "Get your IDs from your Tawk.to Dashboard under `Administration > Channels > Chat Widget`."
)

# Use the IDs from your code as the default values.
# IMPORTANT: Replace these with your actual, active IDs.
default_prop_id = "6871795c3606072bf849ed1e"
default_widget_id = "1ivtk457m"

prop_id = st.sidebar.text_input("Property ID", value=default_prop_id)
widget_id = st.sidebar.text_input("Widget ID", value=default_widget_id)

# --- Main App Content ---

st.title("🚀 Streamlit App with Tawk.to Live Chat")

st.markdown(
    """
    This app demonstrates a reliable method to embed a Tawk.to live chat widget.
    
    ### Why the `st.markdown` method fails:
    The standard `st.markdown(..., unsafe_allow_html=True)` method often fails because Streamlit sandboxes the HTML and can re-render the page, which disrupts external scripts like Tawk.to.
    
    ### The Solution:
    We use `st.components.v1.html` to render a piece of JavaScript. This script then "escapes" its sandboxed iframe and injects the Tawk.to script into the main page's `<head>`. This ensures the script loads correctly and the chat widget can float over all other content as intended.
    
    ---
    """
)


# Embed the Tawk.to component if IDs are provided
if prop_id and widget_id:
    tawk_to_component(property_id=prop_id, widget_id=widget_id)
    st.success("Tawk.to component loaded! Check the bottom-right corner of your screen for the chat widget.")
else:
    st.sidebar.warning("Please enter your Tawk.to IDs to activate the chat widget.")


# Add some example content to the app to show the widget floats over it
st.header("Example App Content")
st.write("Interact with these widgets. The chat button should remain fixed in the corner.")

col1, col2 = st.columns(2)
with col1:
    st.slider("A slider", 0, 100, 50, key="slider1")
    st.text_input("A text input field", key="text1")

with col2:
    st.selectbox("A select box", ["Option 1", "Option 2", "Option 3"], key="select1")
    st.button("A button", key="button1")



####################
## Chat bot Entry ##
####################
#
#def tawk_to_component(property_id: str, widget_id: str):#
#    """
#    A Streamlit component to embed the Tawk.to chat widget.
#
#    This function uses a workaround to inject the Tawk.to script into the main 
#    document's head. This is necessary for the widget to float correctly over 
#    the entire Streamlit app, rather than being confined to a specific component area.
#
#    Args:
#        property_id (str): Your Tawk.to Property ID.
#        widget_id (str): Your Tawk.to Widget ID (e.g., 'default' or a custom ID).
#    """
#    # The JavaScript code that will be executed in the browser.
#    # It creates a script element and adds it to the parent document's head.
#    component_code = f"""
#        <!-- This component is designed to be invisible (height=0) -->
#        <!-- It injects the Tawk.to script into the main Streamlit app document. -->
#        <script type="text/javascript">
#            // This is the standard Tawk.to script
#            var Tawk_API = Tawk_API || {{}}, Tawk_LoadStart = new Date();
#            
#            // We are creating a function that will be executed immediately
#            (function() {{
#                // Check if the Tawk.to script is already present in the parent document
#                const tawkToSrc = 'https://embed.tawk.to/{property_id}/{widget_id}';
#                if (parent.document.querySelector(`script[src="${{tawkToSrc}}"]`)) {{
#                    // If the script is already there, don't add it again.
#                    // This prevents duplication when Streamlit re-runs the script.
#                    return;
#                }}
#
#                // Create a new script element for the Tawk.to widget
#                var s1 = parent.document.createElement("script");
#                s1.async = true;
#                s1.src = tawkToSrc;
#                s1.charset = 'UTF-8';
#                s1.setAttribute('crossorigin', '*');
#                
#                // Find the first script tag in the parent document
#                var s0 = parent.document.getElementsByTagName("script")[0];
#                
#                // Insert the new script element before the first script tag
#                // This is the recommended way to add async scripts.
#                if (s0) {{
#                    s0.parentNode.insertBefore(s1, s0);
#                }} else {{
#                    // If no script tags are found, just append it to the head.
#                    parent.document.head.appendChild(s1);
#                }}
#            }})();
#        </script>
#    """
#    
#    # Use st.components.v1.html to render the JS code
#    # Setting height=0 makes the component invisible.
#    components.html(component_code, height=0)
#
#default_prop_id = "6871795c3606072bf849ed1e"
#default_widget_id = "1ivtk457m"
#
#tawk_to_component(property_id=default_prop_id, widget_id=default_widget_id)
