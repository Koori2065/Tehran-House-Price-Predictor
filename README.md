TehranHousePricePredictor
Overview
This project predicts apartment prices in Tehran, Iran, using machine learning. It utilizes the House_cleaned.csv dataset, which includes features like area, number of rooms, parking, warehouse, elevator, and address. The model employs non-linear regression (cubic polynomial) to estimate prices in USD and IRR, with a focus on simplicity and interpretability for educational purposes.
Dataset

Source: House_cleaned.csv
Features:
Area: Apartment size (square meters)
Room: Number of bedrooms
Parking, Warehouse, Elevator: Binary (0/1) amenities
Address: Location (categorical)
Price: Price in IRR
Price(USD): Price in USD



Methodology

Preprocessing:
Convert boolean features to 0/1.
Encode Address using Target Encoding (mean price per location).
Remove outliers (Area < 5th/95th percentile, Price(USD) < 5th/95th percentile).
Combine features into a weighted feature (Area + 20*Room + 15*Parking + 15*Warehouse + 15*Elevator + 0.1*Address_encoded).
Normalize data using Min-Max scaling (x/max(x)).


Model:
Cubic polynomial regression (y = a*x^3 + b*x^2 + c*x + d) to capture non-linear relationships.
Train-test split: 80% training, 20% testing (randomized with np.random.rand).


Evaluation:
Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score.
Visualizations: Scatter plots with fitted curve.



Installation

Clone the repository:git clone https://github.com/your-username/TehranHousePricePredictor.git


Install dependencies:pip install pandas numpy scikit-learn scipy matplotlib


Place House_cleaned.csv in the project directory.

Usage
Run the script to train the model and predict prices:
python main.py

Example: Predict price for an apartment (100 m², 2 rooms, parking, warehouse, elevator, Saadat Abad):

Edit new_house in main.py for custom inputs.
Outputs price in USD and IRR (1 USD = 30,000 IRR assumed).

Results

Sample metrics (example):
MAE: 0.12 (normalized)
MSE: 0.02 (normalized)
R² Score: ~0.7


Visualizations saved in plots/ folder.

Future Improvements

Test additional non-linear models (e.g., logarithmic or exponential).
Implement One-Hot Encoding for Address to capture location effects better.
Use cross-validation for robust evaluation.
Explore advanced models like Random Forest or Neural Networks.

License
MIT License
Copyright (c) 2025 [Your Name]
Permission is hereby granted, free of charge, to any person obtaining a copy...
