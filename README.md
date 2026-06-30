# Machine-Learning-Project
🧑 Face Recognition — PCA vs LDA with K-NN

A machine learning project implementing and comparing PCA (Eigenfaces) and LDA (Fisherfaces) for face recognition on the ORL (AT&T) Face Dataset, using K-Nearest Neighbors as the classifier.


📌 Project Overview

This project explores dimensionality reduction techniques for face recognition. Both PCA and LDA are implemented from scratch using NumPy (no sklearn decomposition), then compared across different configurations for accuracy, components used, and train/test split strategies.


🗂️ Dataset

ORL Face Dataset — 400 grayscale images of 40 subjects (10 images each), each image 92×112 pixels.

data/
├── s1/   (10 images: 1.pgm … 10.pgm)
├── s2/
⋮
└── s40/

Place data/ in the same directory as the notebook before running.


🔬 What's Covered

1. Data Preparation


Load all 400 images into a matrix D (400 × 10304)
Build label vector y (1–40)
Two train/test splits:

50/50 — odd images for train (5/subject), even for test (5/subject)
70/30 — first 7 images for train, last 3 for test (bonus)





2. PCA — Eigenfaces


Implemented via the surrogate covariance trick (avoids building 10304×10304 matrices)
Alpha parameter controls how much variance is retained
Tested with α ∈ {0.80, 0.85, 0.90, 0.95}


AlphaComponents (r)Accuracy (K=1)0.803695.0%0.855195.0%0.907694.0%0.9511594.0%

3. LDA — Fisherfaces


Implemented via PCA → LDA pipeline (Fisherfaces approach) for efficiency
Reduces to 39 discriminant components (C − 1 for 40 classes)
Accuracy (K=1): 93.0%


4. K-NN Hyperparameter Tuning


Tested K ∈ {1, 3, 5, 7} for both PCA and LDA
K=1 consistently outperforms higher K values for this dataset


5. PCA vs LDA Comparison


PCA (α=0.80 or 0.85) achieves the best accuracy at 95%
LDA achieves 93% with far fewer components (39 vs 36–115)


6. 50/50 vs 70/30 Split (Bonus)


More training data (70/30) generally improves accuracy for PCA
PCA α=0.85: 95.0% → 96.7% with more training data
LDA showed a slight drop, suggesting sensitivity to training set size


7. Binary Classification — Faces vs Non-Faces


Extended to detect whether an image is a face or not
Used binary LDA (1 discriminant vector, since C=2)
Non-face images sourced from scikit-image sample images
Tested with varying numbers of non-face images (10 → 2000)
Accuracy stabilizes at 100% once ≥ 50 non-face samples are used



📊 Results Visualizations

PlotDescriptionpca_accuracy_vs_alpha.pngAccuracy and component count across alpha valuespca_vs_lda.pngSide-by-side accuracy comparison of PCA and LDAknn_tuning.pngEffect of K on accuracy for PCA and LDAsplit_comparison.png50/50 vs 70/30 split accuracy comparisonaccuracy_vs_nonfaces.pngBinary classification accuracy vs. number of non-face imagessuccess_failure_cases.pngSample correct and incorrect predictions


🛠️ Tech Stack

Python · NumPy · scikit-learn (KNN & metrics only) · Pillow · Matplotlib · scikit-image


🚀 How to Run


Download the ORL Face Dataset and place it in a data/ folder
Install dependencies:


bash   pip install numpy matplotlib scikit-learn pillow scikit-image


Open ML.ipynb and run all cells top to bottom



💡 Key Takeaways


PCA with low alpha (0.80–0.85) + K=1-NN gives the best face recognition accuracy on this dataset
LDA is competitive and more compact (39 components vs 36–115), but slightly less accurate here
For binary face/non-face detection, even a small set of non-face images (~50) is enough for the LDA classifier to achieve 100% accuracy
When non-face images greatly outnumber face images, accuracy becomes a misleading metric — F1-score or AUC-ROC are more appropriate
