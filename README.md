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

To gather more information from the data, we can separate it into dimensions, such as gender, to achieve a more comprehensive analysis and identify frequencies and outliers.

Using frequency distribution graphs is possible to notice the following data:
Both male and female income distributions peak in the mid-income range (roughly $40k–$80k), indicating that most customers fall within this bracket.
* The female distribution is more concentrated around the center, suggesting less variability and a higher frequency of middle-income earners.
* The male distribution has a longer right tail, indicating greater variability and the presence of higher-income outliers, which causes a slight right skew.

[Annual Income - Gender](https://github.com/Ktiscar1/Marketing-Group/blob/b5faeef4c7e926afe34477b7dbefd6cd3436093a/Annual_incomme_gender.png)

Using the frequency distribution graph is possible to notice the following data:
Overall overlap:
The two curves overlap a lot, meaning many males and females have similar spending scores.

* Females show a higher peak around 45–55, suggesting a larger proportion of female customers cluster in the mid-to-high spending range.
* Males have a broader, flatter distribution, indicating more variability in spending behavior.
* Males appear slightly more represented at lower spending scores (around 0–25) compared to females.
* Both genders extend into the high spending range (80–100), but females show a slightly stronger presence around 70–80.

* Both males and females are most concentrated between 20 and 45 years, indicating this is the dominant customer age range.
* Female customers: Peak density is around 30–35 years, showing a strong concentration in early adulthood.
* Male customersThe peak is slightly younger (mid-to-late 20s) and the curve is broader, indicating greater age diversity.
* Males extend more into older age groups (50–70+) than females.

[Age - Gender](https://github.com/Ktiscar1/Marketing-Group/blob/33a44d051155ccf3abf487d64e8da25ae6aabcf8/age_gender.png)

[Spending Score - Gender](https://github.com/Ktiscar1/Marketing-Group/blob/1357b24316666f0a79843ef0794b085f78e2925d/spending_score_gender.png)
