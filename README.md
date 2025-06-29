# CHEST X-RAY DISEASES CLASSIFICATION
![alt text](guiDemonstration.png)
*[Link to dataset](https://www.kaggle.com/datasets/ashery/chexpert)

Files in running order:
1. data_preprocessing.ipynb: For data split and testing augmentations.
2. vit-lora.ipynb and contrastive-vit-unsup.ipynb: One script for normal supervised, the other for contrastive learning, both use "vit_base_patch16_224" mocdel and LoRA adapters.
3. test_model.ipynb: For plotting the mean AUROC graph for model checkpoints and determine classification thresholds for each of the 14 classes in the dataset.
4. create_demo_test.ipynb: For extracting some images from the test set to do demonstration.
5. guiDL.py: To run the interface as shown in the image above.


