# A Multimodal Deep Learning Approach for Detection of Opthalmic Biomarkers in 3D Optical Coherence Tomography (OCT) Imagery
This project is done as a part of the Fundamentals of Machine Learning course (ECE 8803) at Georgia Institute of Techology.

# Abstract
This work presents a multimodal framework that leverages a pre-trained vision transformer (ViT), a pre-trained ResNet, and a custom clinical label model to enhance biomarker detection in 3D OCT scans. The framework incorporates imaging data from OCT imagery and clinical data such as central subfield thickness (CST) and best corrected visual clarity (BCVA), which are analyzed independently and in combination. The Grad-CAM model interpretability technique is applied to provide transparency in the decision-making process, aiding clinical validation and adoption. Experimental results on the OLIVES dataset demonstrate that multimodal fusion significantly outperforms individual modalities, achieving state-of-the-art results with a macro F1 score as high as 0.9851. Furthermore, the clinical label model alone highlights the potential to detect certain biomarkers with high precision in the absence of OCT scans, making this framework adaptable to resource-constrained settings.

# Dataset
The Ophthalmic Labels for Investigating Visual Eye Semantics (OLIVES) dataset consists of various forms of data used in the diagnosis of eye diseases, including Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME). This data comes in the form of clinical labels, 2D fundus images, 3D OCT images, and biomarker vectors. The OLIVES dataset is unique in that it contains all the above-mentioned data forms used in diagnosing, and it consists of data from the same patient over the course of a treatment plan [1]. It contains 9408 biomarker- labeled OCT scans collected from 96 patiens and an additional 78185 unlabeled images.

# Contents of the repository
### Directories
- **`labels/`**  
  Includes the original label file as well as the test/train split files we created.
- **`TrainedModels/`**  
  Contains all final trained weights for each model.
### Notebooks
- **`FML_Project_Clinical_Label_Model.ipynb`**  
  Predicts biomarkers from clinical test data (CST and BCVA) only.
- **`FML_Project_ModelEvaluation_GradCAM.ipynb`**  
  Evaluates the trained model and visualizes GradCAM for different models. Does not work for the (non-image) clinical label model.
- **`FML_Project_PreprocessingData.ipynb`**  
  Performs the dataset split and details data preprocessing for image and fusion models.
- **`FML_Project_Resnet.ipynb`**  
  Implements a model using ResNet50 for biomarker predictions.
- **`FML_Project_Resnet_Fusion.ipynb`**  
  The first version of the fusion model, combining ResNet50 and attention mechanisms. Performs worse than the updated version.
- **`FML_Project_Resnet_Fusion_V2.ipynb`**  
  The updated fusion model with a ResNet50 backbone, achieving the best overall performance.
- **`FML_Project_ViT.ipynb`**  
  Uses a Vision Transformer (ViT) for image processing. Includes the fusion model with a ViT backbone.

# Fusion Model Methodology
![alt text](https://github.com/rdharini2001/ECE_8803_Final_Project/blob/main/method.png)

# Results 
![alt text](https://github.com/rdharini2001/ECE_8803_Final_Project/blob/main/ModelPerformanceComparison.png)     

GRAD CAM Visualization of the ResNet50 Fusion Model
![alt text](https://github.com/rdharini2001/ECE_8803_Final_Project/blob/main/GradCam_FusionResNet50.png)     










