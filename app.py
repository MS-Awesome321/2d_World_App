from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from PIL import Image
import plotly.express as px
import io
import base64
import numpy as np
from segmenter import Segmenter
from material import graphene, wte2, hBN
import warnings
import cv2

warnings.simplefilter('ignore', UserWarning)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "2d_World Segmenter"

colors_by_layer = {
    'Monolayer': np.array([0,163,255]), # Blue
    'Bilayer': np.array([29,255,0]), # Green
    'Trilayer': np.array([198,22,22]), # Red
    'Fewlayer': np.array([255,165,0]), # Orange
    'Manylayer': np.array([255,165,0]), # Orange
    'Bluish_layers': np.array([152,7,235]), # Purple
    'Bulk': np.array([152,7,235]), # Purple
    'Dirt': np.array([255, 255, 0]), # Yellow
    'More_bluish_layers': np.array([255, 255, 0]), # Yellow
    'Background': np.array([0, 0, 0]), # Uncolored
}

materials = {
    'graphene': graphene,
    'wte2': wte2,
    'hBN': hBN,
}

def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert("RGBA")
    return img

# -------------------APP LAYOUT--------------------

form = html.Div(
    [
        dbc.Label("Magnification", html_for="magnification", style={'fontWeight':'bold'}),
        dbc.Select(
            [
                {'label': '5x', 'value': 5},
                {'label': '10x', 'value': 10},
                {'label': '20x', 'value': 20},
                {'label': '50x', 'value': 50},
                {'label': '100x', 'value': 100},
            ],
            10,
            id="mag-select", required=True, style={'marginBottom': '2vh', "width": "100%"},
        ),

        dbc.Label("Material", html_for="material", style={'fontWeight':'bold'}),
        dbc.Select(
            [
                {'label': 'Graphene', 'value': 'graphene'},
                {'label': 'WTe2', 'value': 'wte2'},
                {'label': 'hBN', 'value': 'hBN'},
            ],
            "graphene",
            id="mat-select", required=True, style={'marginBottom': '2vh', "width": "100%"},
        ),

        dbc.Label("Shrink Factor", html_for="shrink", style={'fontWeight':'bold'}),
        dbc.Input(value=1, id="shrink", type='number', placeholder=1, required=True, style={'marginBottom': '2vh', "width": "100%"}),
        dbc.Tooltip("Try increasing this in large areas remains unsegmented.", target="shrink"),

        html.Div([
            dbc.Button("Segment", id="segment-again-button", type='submit', style={'margin': 'auto'})
        ], style={'display': 'flex', 'justifyContent':'center'})
    ],
    style={'margin': '5%'}
)

app.layout = html.Div([
    # -------TOP---------
    html.Div([
        html.H1("2d_World Segmenter", style={'fontWeight':'bold'}),
    ], style={"marginTop":"2vh", "marginBottom":"2vh", "width": "100%"}),
    html.Div([
        # -----LEFT SIDE-----
        html.Div([
            html.Div("Overlay Opacity"),
            dcc.Slider(
                id='opacity-slider',
                min=0, max=1, step=0.01, value=0.5,
                marks={0: '0', 0.5: '0.5', 1: '1'},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ),
            dcc.Loading(
                id="loading-component",
                type="default",
                children=[
                    html.Div(id='filename-div', style={'width': '50vw', 'float':'left'}),
                    dcc.Graph(
                        id='output-image-upload',
                        style={'width': '100%'},
                        config={
                            'scrollZoom': True,
                            'displayModeBar': True
                        }
                    ),
                ]
            ),
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select Image')]),
                style={
                    'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                    'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
                },
                multiple=False
            ),
        ], style={'width': '50%', 'float':'left'}),

        # -----RIGHT SIDE-----
        html.Div([
            html.Div([
                html.Div([
                    html.Span(id='hover-data'),
                    html.Br(),
                    html.Span(id='hover-label'),
                ], style={'border':'2px solid black', 'borderRadius':'10px', 'padding':'10px'}),
                form,
            ])
        ], style={'float':'right', 'display':'flex', 'justifyContent':'center', 'width':'50%'}),

        # ------CACHE------
        dcc.Store(id='segmentation-result'),
        dcc.Store(id='raw-segmentation-store'),
    ], style={'display':'flex', 'alignItems':'center',})
], style={'width': "90%", 'margin':'auto', 'height':'90%', 'marginTop': '5%'})


# ------------------------------------------------

@callback(
    Output('output-image-upload', 'figure'),
    Input('upload-image', 'contents'),
    Input('opacity-slider', 'value'),
    Input('segmentation-result', 'data'),
    State('output-image-upload', 'relayoutData'),
    Input('segment-again-button', 'n_clicks'),
)
def update_output(base_content, opacity, result_b64, relayoutData, n_clicks):
    if base_content is not None:
        base_img = parse_image(base_content)
        base_np = np.array(base_img).astype(np.uint8)
        base_np_rgb = base_np[:, :, :3]
        if result_b64 is not None:
            overlay_bytes = base64.b64decode(result_b64)
            overlay_img = Image.open(io.BytesIO(overlay_bytes)).convert("RGB")
            overlay_np = np.array(overlay_img).astype(np.uint8)
            blended = (1 - opacity) * base_np_rgb + opacity * overlay_np
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            fig = px.imshow(blended, height=600)
        else:
            fig = px.imshow(base_np_rgb, height=600)

        fig.update_traces(hovertemplate='x-pos: %{x} <br>y-pos: %{y} <extra></extra>')
        fig.update_layout(
            dragmode='pan', 
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.update_xaxes(fixedrange=False, showticklabels=False)
        fig.update_yaxes(fixedrange=False, showticklabels=False)

        if relayoutData:
            if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
                fig.update_xaxes(range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']])
            if 'yaxis.range[0]' in relayoutData and 'yaxis.range[1]' in relayoutData:
                fig.update_yaxes(range=[relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']])
        return fig
    return {}

@app.callback(
    Output('segmentation-result', 'data'),
    Output('filename-div', 'children'),
    Output('raw-segmentation-store', 'data'),
    Input('segment-again-button', 'n_clicks'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('mat-select', 'value'),
    State('mag-select', 'value'),
    State('shrink', 'value'),
    prevent_initial_call=True
)
def run_segmenter(n_clicks, base_content, filename, material, magnification, shrink):
    if base_content is not None and n_clicks:
        base_img = parse_image(base_content)
        base_np = np.array(base_img).astype(np.uint8)
        base_np_rgb = base_np[:, :, :3]
        base_np_rgb = cv2.resize(base_np_rgb, (int(base_np_rgb.shape[1]/shrink), int(base_np_rgb.shape[0]/shrink)))
        segmenter = Segmenter(base_np_rgb, material=materials[material], colors=colors_by_layer, magnification=int(magnification), max_area=100000000)
        segmenter.process_frame()
        result = segmenter.prettify()
        result = cv2.resize(result, (int(base_np.shape[1]), int(base_np.shape[0])))
        raw_result = segmenter.labelify(shrink)
        # Ensure result is uint8 and RGB
        if result is not None:
            if result.dtype != np.uint8:
                result = result.astype(np.uint8)
            if result.ndim == 2:
                result = np.stack([result]*3, axis=-1)
            img_pil = Image.fromarray(result)
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            result_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            # Store raw_result as JSON-serializable list
            raw_result_list = raw_result.tolist() if raw_result is not None else None
            return result_b64, html.H4(filename), raw_result_list
    return None, "", None

app.clientside_callback(
    """
    function(hoverData, rawSegmentation) {
        const colors_by_layer = {
            'Monolayer': [0,163,255], // Blue
            'Bilayer': [29,255,0], // Green
            'Trilayer': [198,22,22], // Red
            'Fewlayer': [255,165,0], // Orange
            'Manylayer': [255,165,0], // Orange
            'Bluish_layers': [0,0,0], // Uncolored
            'Bulk': [152,7,235], // Purple
            'Dirt': [255, 255, 0], // Yellow
            'More_bluish_layers': [255, 255, 0], // Yellow
            'Background': [0, 0, 0], // Uncolored
        }

        const labelSpan = document.getElementById('hover-label');

        if(hoverData && rawSegmentation && hoverData.points && hoverData.points.length > 0) {
            var x = hoverData.points[0].x;
            var y = hoverData.points[0].y;

            if (y < rawSegmentation.length && x < rawSegmentation[0].length) {
                var label = rawSegmentation[y][x];
                var color = colors_by_layer[label];
                labelSpan.style.color = 'rgb('+color+')';
                return [('Mouse position: x = ' + x + ', y = ' + y), label];
            }
        }
        return ['Upload an image to see segmentation', ''];
    }
    """,
    Output('hover-data', 'children'),
    Output('hover-label', 'children'),
    Input('output-image-upload', 'hoverData'),
    Input('raw-segmentation-store', 'data')
)

if __name__ == '__main__':
    app.run(debug=True)
