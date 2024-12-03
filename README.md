# A Multimodal Deep Learning Approach for Detection of Opthalmic Biomarkers in 3D Optical Coherence Tomography (OCT) Imagery
This project is done as a part of the Fundamentals of Machine Learning course (ECE 8803) at Georgia Institute of Techology.

# Abstract
This work presents a multimodal framework that leverages a pre-trained vision transformer (ViT), a pre-trained ResNet, and a custom clinical label model to enhance biomarker detection in 3D OCT scans. The framework incorporates imaging data from OCT imagery and clinical data such as central subfield thickness (CST) and best corrected visual clarity (BCVA), which are analyzed independently and in combination. The Grad-CAM model interpretability technique is applied to provide transparency in the decision-making process, aiding clinical validation and adoption. Experimental results on the OLIVES dataset demonstrate that multimodal fusion significantly outperforms individual modalities, achieving state-of-the-art results with a macro F1 score as high as 0.9851. Furthermore, the clinical label model alone highlights the potential to detect certain biomarkers with high precision in the absence of OCT scans, making this framework adaptable to resource-constrained settings.

# Dataset
The Ophthalmic Labels for Investigating Visual Eye Semantics (OLIVES) dataset consists of various forms of data used in the diagnosis of eye diseases, including Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME). This data comes in the form of clinical labels, 2D fundus images, 3D OCT images, and biomarker vectors. The OLIVES dataset is unique in that it contains all the above-mentioned data forms used in diagnosing, and it consists of data from the same patient over the course of a treatment plan [1]. It contains 9408 biomarker- labeled OCT scans collected from 96 patiens and an additional 78185 unlabeled images.

# Overall Methodology

![alt text](https://github.com/rdharini2001/ECE_8803_Final_Project/blob/main/method.png)

# Results 
![alt text](https://github.com/rdharini2001/ECE_8803_Final_Project/blob/main/ModelPerformanceComparison.png)     

GRAD CAM Visualization of the ResNet50 Fusion Model
![alt text](https://github.com/rdharini2001/ECE_8803_Final_Project/blob/main/GradCam_FusionResNet50.png)     


# Contents of the repository
labels/ - Includes the original label file as well as the test/train split files we created.
TrainedModels/ - Includes all final trained weights for each model.
FML_Project_Clinical_Label_Model.ipynb - Predicts biomarkers from clinical test data (CST and BCVA) only.
FML_Project_ModelEvaluation_GradCAM.ipynb - Can be used to evaluate the trained model as well as visualize a GradCAM for different models. Doesn't work for the (non-image) clinical label model.
FML_Project_PreprocessingData.ipynb - Performs the split of the dataset. Also shows the data preprocessing used in the different image/fusion models.
FML_Project_Resnet.ipynb - Model using a ResNet50 to perform the biomarker predicitons.
FML_Project_Resnet_Fusion.ipynb - First version of the fusion model using a ResNet50 model and attention mechanisms. Worse performance than it's newer model.
FML_Project_Resnet_Fusion_V2.ipynb - Updated fusion model with a ResNet50 backbone. Showed the best overall performance.
FML_Project_ViT.ipynb - Model using a vision transformer for image processing. This also includes the fusion model using that same vision transformer as its backbone.








