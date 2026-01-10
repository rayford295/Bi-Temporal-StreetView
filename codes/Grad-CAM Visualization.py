# ==============================================================================
# Multi-Backbone Grad-CAM Visualization for ConvNeXt and Swin
# ==============================================================================

import torch
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from google.colab import files  # For downloading images in Colab

# ==============================================================================
# 1. Device Setup
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. Load Pretrained Models
# ==============================================================================
model_names = [
    "convnext_tiny",
    "convnext_base",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
]

models = {
    name: timm.create_model(name, pretrained=True).eval().to(device)
    for name in model_names
}

# ==============================================================================
# 3. Select Target Layers for Grad-CAM
# ==============================================================================
target_layers = {
    "convnext_tiny": models["convnext_tiny"].stages[3],  # ConvNeXt final stage
    "convnext_base": models["convnext_base"].stages[3],  # ConvNeXt final stage
    "swin_small_patch4_window7_224": models["swin_small_patch4_window7_224"].layers[-1].blocks[-1].norm1,  # Swin
    "swin_base_patch4_window7_224": models["swin_base_patch4_window7_224"].layers[-1].blocks[-1].norm1,    # Swin
}

# ==============================================================================
# 4. Image Preprocessing
# ==============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    """Load an image and convert it into a normalized input tensor."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    return image, input_tensor

# ==============================================================================
# 5. Grad-CAM Implementation
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # NOTE: register_backward_hook is deprecated but kept here to match your original code behavior.
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for a given input.
        If class_idx is None, the predicted class is used.
        """
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()
        output[:, class_idx].backward()

        gradients = self.gradients
        activations = self.activations

        # If activations are (B, L, C), reshape into (B, C, H, W)
        if len(activations.shape) == 3:
            B, L, C = activations.shape
            H = W = int(np.sqrt(L))
            activations = activations.permute(0, 2, 1).contiguous().view(B, C, H, W)
            gradients = gradients.permute(0, 2, 1).contiguous().view(B, C, H, W)

        weights = gradients.mean(dim=[2, 3], keepdim=True)
        grad_cam = (weights * activations).sum(dim=1, keepdim=True)

        grad_cam = torch.clamp(grad_cam, min=0)
        grad_cam = grad_cam / (grad_cam.max() + 1e-8)

        return grad_cam.squeeze().cpu().numpy()

# ==============================================================================
# 6. Overlay Heatmap on Original Image
# ==============================================================================
def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay a Grad-CAM heatmap onto the input image.
    """
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_np = np.array(image)
    superimposed_img = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    return Image.fromarray(superimposed_img)

# ==============================================================================
# 7. Run Grad-CAM for All Models
# ==============================================================================
image_path = "/content/3754051524846539_2024.png"  # Replace with your image path
image, input_tensor = preprocess_image(image_path)

for model_name, model in models.items():
    print(f"Processing {model_name}...")

    grad_cam = GradCAM(model, target_layers[model_name])
    heatmap = grad_cam.generate_heatmap(input_tensor)

    result_img = overlay_heatmap(image, heatmap)

    output_path = f"/content/{model_name}_gradcam.png"
    result_img.save(output_path)

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(result_img)
    axes[1].set_title(f"Grad-CAM: {model_name}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Download in Colab
    files.download(output_path)

print("All models processed!")
