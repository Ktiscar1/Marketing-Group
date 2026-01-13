# Marketing-Group
The analysis focused on segmenting a specific group of mall customers using the KMeans unsupervised machine learning algorithm. Univariate, bivariate, and multivariate clusters were identified and analyzed using summary statistics to gain insights into customer behavior and determine the most valuable segment for targeted marketing strategies.

The objective of this problem is to identify the most important shopping groups by analyzing customer characteristics such as income level, age, and mall shopping score. By examining patterns and similarities within these variables, the task aims to determine the ideal number of distinct customer groups that best represent the underlying structure of the data. Each customer is then assigned a clear group label, enabling better understanding of shopping behaviors and supporting targeted marketing strategies, personalized services, and data-driven business decisions.

# Approach
1. Perform EDA
2. Use K-means clustering algorithm to create segments
3. Summarize Statics on the clusters

# Exploratory Data Analysis
Focusing on variable at a time allows use to find patterns within that single variable.

We can notice that both columns for "Age" and "Annual Income (k$)" present a positive skew where most of the data is concentrated in the lower to middle income ranges.

![Histogram for Annual Income](https://github.com/Ktiscar1/Marketing-Group/blob/36365d96f4da7aa2c28bb08c9d07a06d12a279f6/annual_income_analysis_uni.png)

![Histogram for Age](https://github.com/Ktiscar1/Marketing-Group/blob/f7e8497551b92cf8e570dfd1c8f746aec183db78/age_uni.png)

We can observe that the column "Spending Score (1-100)" contains a symmetrical Distribution where the data appears fairly balanced around the center (around a spending score of 50), with similar frequencies on both the lower and higher sides.

![Histogram for Spending Score](https://github.com/Ktiscar1/Marketing-Group/blob/4894b1952a3b26129fa632f2160fa6ed77b7b68d/Spending_Score_Uni.png)
