## Customer Segmentation using K-Means Clustering

## 📌 Objective
This project applies **K-Means Clustering** to segment mall customers based on their demographic and spending behavior.  
The optimal number of clusters is determined using the **Elbow Method** and validated with the **Silhouette Score**.  
Results are visualized using **PCA** for better interpretability.

---
## Features Used for Clustering:

Age

Annual Income (k$)

Spending Score (1-100)



---

## 🛠 Steps Performed

1. Import Libraries – pandas, matplotlib, seaborn, sklearn, kagglehub.


2. Load Dataset – Used kagglehub to download and load the dataset.


3. Feature Selection – Chose only numerical columns for clustering.


4. Data Scaling – Standardized features with StandardScaler.


5. Elbow Method – Determined the optimal number of clusters (K).


6. K-Means Training – Applied K-Means clustering with chosen K value.


7. Evaluation – Calculated Silhouette Score to measure cluster quality.


8. Visualization – Reduced dimensions to 2D using PCA and plotted clusters.




---

## 📊 Results

Optimal K: 5 (based on Elbow Method)

Silhouette Score: ~0.55 (score may slightly vary on different runs)

## Outputs Generated:

<img width="720" height="1640" alt="1000020064" src="https://github.com/user-attachments/assets/0f63db79-2bb5-44d2-ad32-81e726f85ec2" />


Elbow Method plot

PCA-based cluster scatter plot




---
