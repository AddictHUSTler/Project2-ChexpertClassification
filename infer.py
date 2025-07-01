import argparse
import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import pandas as pd
import os

# --- Model and LoRA Configuration ---

# Define the LoRA configuration, consistent with the training scripts.
LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=256,
    target_modules=["qkv", "proj"],
    lora_dropout=0.1,
    bias="none"
)

# Define the model class for the contrastive pre-trained ViT.
# This class includes the ViT backbone and a final classifier layer.
class FineTuningViT(nn.Module):
    def __init__(self, lora_config, num_classes=14, drop_rate=0.1):
        super(FineTuningViT, self).__init__()
        # Create a ViT model without pre-trained weights and no classifier head.
        backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0, # num_classes=0 returns features before the head
            drop_rate=drop_rate
        )
        # Apply LoRA to the backbone
        self.backbone = get_peft_model(backbone, lora_config)
        # Add the final classification layer
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        """The forward pass returns the logits from the classifier."""
        features = self.backbone(x)
        return self.classifier(features)

# --- Main Script ---

# List of condition names corresponding to the model output.
CONDITIONS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
]

# Dictionary to hold the loaded models.
models = dict()
CHECKPOINT_DIR = ''
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Define the names of the ViT models to be loaded.
vit_model_names = [
    'ViT-LoRA-U0', 'ViT-LoRA-U1',
    'ViT-LoRA-contrastive-U0', 'ViT-LoRA-contrastive-U1'
]

# Load each ViT model and its state dictionary.
for model_name in vit_model_names:
    model_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint not found at '{model_path}'. Skipping this model.")
        continue

    print(f"Loading model: {model_name}...")
    # The contrastive models have a different architecture wrapper.
    if 'contrastive' in model_name:
        model = FineTuningViT(lora_config=LORA_CONFIG, num_classes=14)
    else:
        # Standard ViT model with LoRA applied.
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=14)
        model = get_peft_model(model, LORA_CONFIG)

    # Load the saved weights into the model.
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    models[model_name] = model
    print(f"Successfully loaded {model_name}.")


# Image transformations for inference.
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.506, 0.506, 0.506], std=[0.287, 0.287, 0.287]),
    ToTensorV2()
])

# --- Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Inference script for CheXpert's Diseases Classification using ViT models. "
                "Ensure that the U-version of the model matches the dataset "
                "(e.g., U1 models use images from 'test/u1')."
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help=f"Model name to use. Available models: {', '.join(models.keys())}"
)
parser.add_argument(
    "--image_path",
    type=str,
    required=True,
    help="Path to the input image. For accurate label comparison, please use images from the 'test' folder. "
         "Use forward slashes ('/') for the path."
)

args = parser.parse_args()

# --- Inference ---
if args.model_name not in models:
    print(f"Error: Model '{args.model_name}' is not available. Please choose from: {', '.join(models.keys())}")
else:
    # Load and transform the image
    try:
        image = Image.open(args.image_path).convert('RGB')
        image_tensor = transforms(image=np.array(image))['image'].to(DEVICE)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{args.image_path}'")
        exit()

    # Select the model
    selected_model = models[args.model_name]

    # --- Prediction and Label Comparison ---
    # Determine the uncertainty version ('u0' or 'u1') from the model name
    u_version = args.model_name.split('-')[-1].lower()

    # Load the corresponding ground truth labels and thresholds
    labels_path = f'test/{u_version}/{u_version}_test.csv'
    thresholds_path = os.path.join(CHECKPOINT_DIR, f'{args.model_name}_best_tuned.csv')

    if not os.path.exists(labels_path) or not os.path.exists(thresholds_path):
        print(f"Error: Required file not found. Check paths for labels ('{labels_path}') and thresholds ('{thresholds_path}').")
    else:
        labels_df = pd.read_csv(labels_path, index_col=0)
        # Standardize the image path to match the index format in the CSV
        image_index = 'chexpert' + args.image_path.split('chexpert', 1)[-1]

        if image_index not in labels_df.index:
            print(f"Error: Image path '{args.image_path}' not found in the labels file '{labels_path}'.")
        else:
            image_label = labels_df.loc[image_index]
            positive_labels = image_label[image_label == 1]

            thresholds = pd.read_csv(thresholds_path)['threshold'].values

            # Perform inference
            with torch.no_grad():
                results = selected_model(image_tensor.unsqueeze(0))
                probabilities = torch.sigmoid(results).squeeze().cpu().detach().numpy()

            # Identify predicted labels based on the tuned thresholds
            pred_positive_indices = np.where(probabilities > thresholds)[0]
            predicted_labels_str = ', '.join([CONDITIONS[i] for i in pred_positive_indices]) if len(pred_positive_indices) > 0 else "None"
            true_labels_str = ', '.join(positive_labels.index) if not positive_labels.empty else "None"

            # Print results
            print("\n--- Inference Results ---")
            print(f"Model: {args.model_name}")
            print(f"Image: {args.image_path}")
            print(f"Predicted labels: {predicted_labels_str}")
            print(f"True labels:      {true_labels_str}")
            print("-------------------------\n")