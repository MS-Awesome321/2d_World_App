import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import streamlit.components.v1 as components
import io

st.set_page_config(layout="wide")
st.title('2d-World Segmenter App')

left_cont, right_cont = st.columns([3, 2], gap='large')

# saco = "test"
# html_string = f'''
#         <div id="res"></div>
#         <script type="text/javascript">
#             let mystr = "{saco}";
#             document.querySelector("#res").innerHTML = mystr;
#             alert(mystr);
#         </script>
#     '''
# components.html(html_string)

with left_cont:
    uploaded_file = left_cont.file_uploader("Upload image from microscope:", type=['jpeg', 'jpg', 'png'])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        st.session_state['img'] = img

        # Only create the figure once and store it in session_state
        if 'fig' not in st.session_state:
            st.session_state['fig'] = px.imshow(img, height=600, binary_string=True)

        left_cont.plotly_chart(st.session_state['fig'])

injected_script = """
    <script>
        var tspans = document.querySelectorAll('tspan');

        function printTspans() {
            console.log('hello');
            console.log(tspans);
        }

        document.addEventListener('mousemove', printTspans);
    </script>
"""
components.html(injected_script)