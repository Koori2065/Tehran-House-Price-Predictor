Customer Segmentation Project



Overview



This project performs customer segmentation using the Customer.csv dataset. It employs unsupervised clustering algorithms (K-Means, Hierarchical Clustering, and DBSCAN) to group customers based on demographic and behavioral features. The analysis is designed for educational purposes, demonstrating data preprocessing, clustering, and visualization using Python and scikit-learn.





Dataset
Source: Customer.csv
Features: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100).
Target: None (unsupervised learning; clusters are derived from features).
Models








K-Means: 5 clusters, best overall performance.
Hierarchical Clustering: With 'complete' linkage, forming clear segments.
DBSCAN: Density-based clustering (not suitable for this dataset; many noise points).








Preprocessing: Gender encoded (Male=0, Female=1), features standardized with StandardScaler.
Visualization: 3D scatter plots for cluster comparison.








Evaluation
Metrics: Visual inspection via 3D plots; K-Means and Hierarchical ('complete') yield balanced clusters, while DBSCAN identifies excessive noise.




Clone the repository
textgit clone https://github.com/Koori2065/CustomerSegmentation.git






License
MIT License
Copyright (c) 2025 [Kourosh Asadi]
Permission is hereby granted, free of charge, to any person obtaining a copy
