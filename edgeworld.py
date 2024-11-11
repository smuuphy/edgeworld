# Import Libraries
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from matplotlib.colors import to_rgba
from io import BytesIO
from PIL import Image
import cv2

# Set up Streamlit page
st.set_page_config(page_title="Exotic Material Art Generator with Still Images and Animations")
st.title("Exotic Material Art Generator with Still Images and Animations")
st.write("""
This app generates high-resolution still images and animations based on exotic materials with higher-dimensional influences.
Upload an image to blend with the generated shapes, choose still images or animations, and apply unique styles for creative effects.
""")

# Exotic Material Base Class and Specific Materials
class ExoticMaterial:
    def __init__(self, reflectivity=0.0, emissivity=0.0, transparency=0.0, curvature_effect=0.0):
        self.reflectivity = reflectivity
        self.emissivity = emissivity
        self.transparency = transparency
        self.curvature_effect = curvature_effect

    def generate_texture(self, size):
        raise NotImplementedError("Each material must implement its own texture generation.")

class DarkMatter(ExoticMaterial):
    def __init__(self):
        super().__init__(reflectivity=0.1, emissivity=-0.5, transparency=0.9, curvature_effect=1.0)

    def generate_texture(self, size):
        texture = np.clip(np.random.normal(0.1, 0.05, (size, size)), 0, 1)
        return texture

class CalabiYauMaterial(ExoticMaterial):
    def __init__(self):
        super().__init__(reflectivity=0.5, emissivity=0.6, transparency=0.4, curvature_effect=1.2)

    def generate_texture(self, size):
        x = np.linspace(0, np.pi, size)
        y = np.linspace(0, np.pi, size)
        X, Y = np.meshgrid(x, y)
        texture = np.abs(np.sin(X * Y)) * self.curvature_effect
        return np.clip(texture, 0, 1)

class TesseractMetal(ExoticMaterial):
    def __init__(self):
        super().__init__(reflectivity=0.9, emissivity=0.1, transparency=0.3, curvature_effect=1.5)

    def generate_texture(self, size):
        texture = np.clip(np.random.normal(0.8, 0.2, (size, size)), 0, 1)
        return np.clip(texture * self.reflectivity, 0, 1)

# Define Shapes and ShapeCluster
class BaseShape:
    def __init__(self, position, size, material):
        self.position = position
        self.size = size
        self.material = material

    def draw(self, ax, color=(1, 1, 1), alpha=1.0):
        raise NotImplementedError("Each shape subclass must implement its own draw method.")

class Square(BaseShape):
    def draw(self, ax, color=(1, 1, 1), alpha=1.0):
        x, y = self.position
        ax.add_patch(plt.Rectangle((x, y), self.size, self.size, color=color, alpha=alpha))

class ShapeCluster:
    def __init__(self, shape_type, count, spacing, base_size, material):
        self.shapes = [Square((i * spacing, i * spacing), base_size, material) for i in range(count)]

    def draw(self, ax):
        for shape in self.shapes:
            color = to_rgba((1 - shape.material.transparency, 1 - shape.material.transparency, 1))
            shape.draw(ax, color=color, alpha=1 - shape.material.transparency)

# Lighting Model
class LightingModel:
    def __init__(self, light_direction=(1, 1, 1), intensity=1.0):
        self.light_direction = np.array(light_direction) / np.linalg.norm(light_direction)
        self.intensity = intensity

    def calculate_lighting(self, position, normal, material):
        dot_product = np.dot(self.light_direction, normal)
        brightness = self.intensity * max(dot_product, 0)
        brightness = brightness * material.reflectivity + material.emissivity
        brightness *= (1 - material.transparency)
        return np.clip(brightness, 0, 1)

# Image Blending
def blend_with_image(base_image, shapes_image, blend_mode="alpha", blend_strength=0.5):
    base_image = np.array(base_image.resize(shapes_image.size))
    shapes_image = np.array(shapes_image)
    
    if blend_mode == "alpha":
        blended = cv2.addWeighted(base_image, 1 - blend_strength, shapes_image, blend_strength, 0)
    else:
        blended = cv2.multiply(base_image, shapes_image)
        
    return Image.fromarray(blended)

# Animation Function (GIF only)
def animate_radiance(shape_cluster, frames=50, interval=100):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis("off")

    def update(frame):
        ax.clear()
        ax.set_xlim(-5, 20)
        ax.set_ylim(-5, 20)
        for shape in shape_cluster.shapes:
            x, y = shape.position
            shape.position = (x + 0.2 * np.sin(frame / 10), y + 0.2 * np.cos(frame / 10))
            shape.draw(ax, color=(0.5, 0.5, 0.8), alpha=0.6)
    return FuncAnimation(fig, update, frames=frames, interval=interval)

# Streamlit UI
st.subheader("Select Exotic Material")
material_options = {"Dark Matter": DarkMatter(), "Calabi-Yau Material": CalabiYauMaterial(), "Tesseract Metal": TesseractMetal()}
selected_material_name = st.selectbox("Choose Material", list(material_options.keys()))
selected_material = material_options[selected_material_name]

st.subheader("Upload Image for Blending")
uploaded_image = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

st.subheader("Configure Shape Cluster")
shape_count = st.slider("Number of Shapes", 1, 10, 5)
spacing = st.slider("Spacing Between Shapes", 1, 5, 2)
base_size = st.slider("Base Size of Shapes", 1, 10, 3)

st.subheader("Configure Blending")
blend_mode = st.selectbox("Blend Mode", ["alpha", "multiply"])
blend_strength = st.slider("Blend Strength", 0.0, 1.0, 0.5)

# Choose between High-Resolution Still Image and Animation
output_type = st.selectbox("Choose Output Type", ["Still Image", "Animation"])

# Render the shape cluster
shape_cluster = ShapeCluster("Square", shape_count, spacing, base_size, selected_material)

# Output High-Resolution Still Image
if output_type == "Still Image":
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_aspect('equal')
    ax.axis("off")
    shape_cluster.draw(ax)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    shapes_image = Image.open(buf)

    if uploaded_image:
        base_image = Image.open(uploaded_image)
        blended_image = blend_with_image(base_image, shapes_image, blend_mode=blend_mode, blend_strength=blend_strength)
        st.image(blended_image, caption="Blended High-Resolution Still Image", use_column_width=True)

    buf.seek(0)
    st.download_button(label="Download High-Resolution Image", data=buf, file_name="high_res_image.png", mime="image/png")

# Generate and Display Animation as GIF
else:
    animation = animate_radiance(shape_cluster)
    buf = BytesIO()
    animation.save(buf, format="gif", writer=PillowWriter(fps=20))
    buf.seek(0)
    st.image(buf, format="gif")
    st.download_button(label="Download Animation as GIF", data=buf, file_name="animation.gif", mime="image/gif")
