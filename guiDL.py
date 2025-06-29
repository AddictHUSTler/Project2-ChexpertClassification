import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import numpy as np
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
import timm
from peft import LoraConfig, get_peft_model

script_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(script_dir, '')


# --- Model Architectures ---
class FineTuningViT(nn.Module):
    def __init__(self, lora_config, num_classes=14, drop_rate=0.1):
        super(FineTuningViT, self).__init__()
        backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, drop_rate=drop_rate)
        self.backbone = get_peft_model(backbone, lora_config)
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)
    def forward(self, x):
        return self.classifier(self.backbone(x))

def get_model_dict():
    """
    Loads all available ViT models from the 'checkpoint' directory.
    """
    models = dict()
    DEVICE = 'cpu'
    LORA_CONFIG = LoraConfig(r=64, lora_alpha=256, target_modules=["qkv", "proj"], lora_dropout=0.1, bias="none")
    VIT_DROPOUT = 0.1
    vit_model_files = ['ViT-LoRA-U0', 'ViT-LoRA-U1', 'ViT-LoRA-contrastive-U0', 'ViT-LoRA-contrastive-U1']

    for model_name in vit_model_files:
        full_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pth")
        if not os.path.exists(full_path):
            print(f"Warning: Model file '{full_path}' not found. Skipping.")
            continue
        try:
            print(f"Loading {model_name}...")
            if 'contrastive' in model_name:
                model = FineTuningViT(lora_config=LORA_CONFIG, num_classes=14, drop_rate=VIT_DROPOUT)
            else: 
                base_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=14, drop_rate=VIT_DROPOUT)
                model = get_peft_model(base_model, LORA_CONFIG)
            
            model.load_state_dict(torch.load(full_path, map_location=DEVICE))
            models[model_name] = model
            print(f"Successfully loaded {model_name}.")

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    return models

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chest X-ray Disease Classification")
        self.setGeometry(100, 100, 1200, 800)
        
        main_layout = QHBoxLayout()
        central_widget = QVBoxLayout()

        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo_box = QComboBox()
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.model_combo_box)
        central_widget.addLayout(model_selection_layout)

        self.image_label = QLabel("Upload an image to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont('Arial', 20))
        self.image_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0; border-radius: 10px;")
        central_widget.addWidget(self.image_label, 5)

        button_layout = QHBoxLayout()
        upload_button = QPushButton("Upload Image")
        report_button = QPushButton("Predict")
        button_layout.addWidget(upload_button)
        button_layout.addWidget(report_button)
        central_widget.addLayout(button_layout)
        upload_button.clicked.connect(self.upload_image)
        report_button.clicked.connect(self.predict)
        main_layout.addLayout(central_widget, 4)

        output_panel = self.setup_output_panel()
        main_layout.addLayout(output_panel, 3)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.image_path = None
        self.model_dict = get_model_dict()
        if not self.model_dict:
             QMessageBox.critical(self, "Model Loading Error", f"No models found in the '{CHECKPOINT_DIR}' directory.")
             sys.exit(1)
        self.model_combo_box.addItems(["None"] + sorted(self.model_dict.keys()))
        self.model_combo_box.currentTextChanged.connect(self.on_model_change)

    def setup_output_panel(self):
        output_panel = QVBoxLayout()
        self.output_title = QLabel("Analysis Results")
        self.output_title.setFont(QFont('Arial', 18, QFont.Bold))
        output_panel.addWidget(self.output_title)
        
        self.conditions = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        
        self.progress_bars = {}
        self.threshold_labels = {} 

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Condition</b>"))
        header_layout.addStretch(1)
        header_layout.addWidget(QLabel("<b>Confidence</b>"))
        header_layout.addStretch(1)
        header_layout.addWidget(QLabel("<b>Threshold</b>"))
        output_panel.addLayout(header_layout)

        for condition in self.conditions:
            row_layout = QHBoxLayout()
            label = QLabel(condition)
            label.setFixedWidth(180)
            progress = QProgressBar()
            progress.setValue(0)
            progress.setFormat("%p%")
            progress.setStyleSheet("QProgressBar { text-align: center; } QProgressBar::chunk { background-color: #3498db; }")
            threshold_label = QLabel("(N/A)")
            threshold_label.setFixedWidth(60)
            threshold_label.setAlignment(Qt.AlignCenter)
            row_layout.addWidget(label)
            row_layout.addWidget(progress)
            row_layout.addWidget(threshold_label)
            output_panel.addLayout(row_layout)
            self.progress_bars[condition] = progress
            self.threshold_labels[condition] = threshold_label

        self.predicted_text = QTextEdit()
        self.predicted_text.setReadOnly(True)
        self.true_text = QTextEdit()
        self.true_text.setReadOnly(True)
        output_panel.addSpacing(20)
        output_panel.addWidget(QLabel("<b>Predicted Diseases:</b>"))
        output_panel.addWidget(self.predicted_text)
        output_panel.addWidget(QLabel("<b>True Diseases (if in test set):</b>"))
        output_panel.addWidget(self.true_text)
        return output_panel

    def on_model_change(self, model_name):
        if not model_name or model_name == "None":
            self.output_title.setText("Analysis Results")
            for condition in self.conditions: self.threshold_labels[condition].setText("(N/A)")
            return
            
        self.output_title.setText(f"Analysis Results for {model_name}")
        
        threshold_filename = f"{model_name}_best_tuned.csv"
        threshold_path = os.path.join(CHECKPOINT_DIR, threshold_filename)

        if not os.path.exists(threshold_path):
            QMessageBox.warning(self, "Thresholds Not Found", f"Threshold file not found:\n{threshold_filename}\nPredictions will use a default of 0.5.")
            for condition in self.conditions: self.threshold_labels[condition].setText("(0.50)")
            return

        try:
            thresholds_df = pd.read_csv(threshold_path)
            thresholds = pd.Series(thresholds_df.threshold.values, index=thresholds_df['class']).to_dict()
            for condition in self.conditions:
                thresh_val = thresholds.get(condition, 0.5)
                self.threshold_labels[condition].setText(f"({thresh_val:.2f})")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read or parse threshold file: {e}")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Chest X-ray", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.true_text.clear()

    def predict(self):
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please upload an image first.")
            return

        selected_model_name = self.model_combo_box.currentText()
        if selected_model_name == 'None':
            QMessageBox.warning(self, "Error", "Please select a model first.")
            return

        threshold_filename = f"{selected_model_name}_best_tuned.csv"
        threshold_path = os.path.join(CHECKPOINT_DIR, threshold_filename)

        if not os.path.exists(threshold_path):
            QMessageBox.critical(self, "Error", f"Optimized threshold file not found:\n{threshold_filename}\nPlease run the final F1-tuning script to generate it.")
            return

        try:
            thresholds_df = pd.read_csv(threshold_path)
            thresholds = pd.Series(thresholds_df.threshold.values, index=thresholds_df['class']).to_dict()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read or parse threshold file: {e}")
            return

        image = Image.open(self.image_path).convert('RGB')
        image_tensor = transforms(image=np.array(image))['image']
        model = self.model_dict[selected_model_name]
        
        with torch.no_grad():
            probabilities = nn.Sigmoid()(model(image_tensor.unsqueeze(0)))

        predicted_labels_list = []
        for i, condition in enumerate(self.conditions):
            prob = probabilities[0, i].item()
            threshold = thresholds.get(condition, 0.5)
            if prob > threshold:
                predicted_labels_list.append(condition)
            self.progress_bars[condition].setValue(int(prob * 100))
            self.threshold_labels[condition].setText(f"({threshold:.2f})")

        self.predicted_text.setText(', '.join(predicted_labels_list) if predicted_labels_list else "No Finding")
        self.find_true_labels(selected_model_name)

    def find_true_labels(self, model_name):
        u_version = model_name[-1]
        test_csv_path = os.path.join(script_dir, 'test', f'u{u_version}', f'u{u_version}_test.csv')

        if not os.path.exists(test_csv_path):
            self.true_text.setText("Test set CSV not found.")
            return
            
        labels_df = pd.read_csv(test_csv_path, index_col=0)
        try:
            image_index = 'chexpert/' + self.image_path.split('chexpert/', 1)[1].replace(os.path.sep, '/')
            if image_index in labels_df.index:
                image_label = labels_df.loc[image_index]
                positive_labels = image_label[image_label == 1].index.tolist()
                self.true_text.setText(', '.join(positive_labels) if positive_labels else "No Finding")
            else:
                self.true_text.setText("Image not found in test set.")
        except (IndexError, KeyError):
            self.true_text.setText("Image not from the test set or path is incorrect.")

transforms = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.506, 0.506, 0.506], std=[0.287, 0.287, 0.287]), ToTensorV2()])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
