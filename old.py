import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_plotly_events import plotly_events as pe
import io
from segmenter import Segmenter
from material import graphene

st.set_page_config(layout="wide")
st.title('2d-World Segmenter App')

left_cont, right_cont = st.columns([3, 2], gap='large')

result_cont = None
uploaded_file = None
img = None

with left_cont:
    uploaded_file = st.file_uploader("Upload image from microscope:", type=['jpeg', 'jpg', 'png'])
    result_cont = st.empty()
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(bytes_data))
        fig = px.imshow(img, height=600, binary_string=True)
        selected_points = pe(fig, click_event=True, hover_event=False)
    else:
        selected_points = None

left_cont.write(selected_points)

number_by_layer = {
    'bg': 0,
    'monolayer': 1,
    'bilayer': 2,
    'trilayer': 3,
    'fewlayer': 3,
    'manylayer': 3,
    'bluish_layers':3,
    'more_bluish_layers':3,
    'bulk': 3,
    'dirt': 3,
}

def segment_img(img):
    input_to_segmenter = np.array(img)
    segmenter = Segmenter(input_to_segmenter, graphene, numbers=number_by_layer, magnification=10)
    segmenter.process_frame()
    result = segmenter.numberify()
    result_cont.empty()
    overlay = input_to_segmenter.copy()
    if result.ndim == 2:
        import matplotlib
        cmap = matplotlib.cm.get_cmap('jet', np.max(result)+1)
        color_mask = (cmap(result)[:, :, :3] * 255).astype(np.uint8)
        alpha = 0.4
        overlay = (alpha * color_mask + (1 - alpha) * overlay).astype(np.uint8)
        
        # Create label map for hover
        label_map = {v: k for k, v in number_by_layer.items()}
        label_array = np.vectorize(lambda x: label_map.get(x, str(x)))(result)
    else:
        label_array = None

    fig_overlay = px.imshow(
        overlay, 
        height=600,
        labels=dict(x="x", y="y", color="Layer Type")
    )
    # Set hovertemplate to show label
    result_cont.plotly_chart(fig_overlay, use_container_width=False)
   

with right_cont:
    if uploaded_file is not None:
        if st.button("Segment Image"):
            segment_img(img)