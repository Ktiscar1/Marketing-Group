# Marketing-Group
The analysis focused on segmenting a specific group of mall customers using the KMeans unsupervised machine learning algorithm. Univariate, bivariate, and multivariate clusters were identified and analyzed using summary statistics to gain insights into customer behavior and determine the most valuable segment for targeted marketing strategies.

The objective of this problem is to identify the most important shopping groups by analyzing customer characteristics such as income level, age, and mall shopping score. By examining patterns and similarities within these variables, the task aims to determine the ideal number of distinct customer groups that best represent the underlying structure of the data. Each customer is then assigned a clear group label, enabling better understanding of shopping behaviors and supporting targeted marketing strategies, personalized services, and data-driven business decisions.

# Approach
1. Perform EDA
2. Use K-means clustering algorithm to create segments
3. Summarize Statics on the clusters

# Exploratory Data Analysis (Univariate & Bivariate)

# Univariate Analysis
Focusing on variable at a time allows use to find patterns within that single variable.

We can notice that both columns for "Age" and "Annual Income (k$)" present a positive skew where most of the data is concentrated in the lower to middle income ranges.

![Histogram for Annual Income](https://github.com/Ktiscar1/Marketing-Group/blob/36365d96f4da7aa2c28bb08c9d07a06d12a279f6/annual_income_analysis_uni.png)

![Histogram for Age](https://github.com/Ktiscar1/Marketing-Group/blob/f7e8497551b92cf8e570dfd1c8f746aec183db78/age_uni.png)
The symmetrical distribution can be observed with the column "Spending Score (1-100)," where the data appears fairly balanced around the center (around a spending score of 50), with similar frequencies on both the lower and higher sides.

![Histogram for Spending Score](https://github.com/Ktiscar1/Marketing-Group/blob/4894b1952a3b26129fa632f2160fa6ed77b7b68d/Spending_Score_Uni.png)

To gather more information from the data, this information can be separated it into dimensions, such as gender, to achieve a more comprehensive analysis and identify frequencies and outliers.

Using frequency distribution graphs is possible to notice the following data:
Both male and female income distributions peak in the mid-income range (roughly $40k–$80k), indicating that most customers fall within this bracket.
* The female distribution is more concentrated around the center, suggesting less variability and a higher frequency of middle-income earners.
* The male distribution has a longer right tail, indicating greater variability and the presence of higher-income outliers, which causes a slight right skew.

![Annual Income - Gender](https://github.com/Ktiscar1/Marketing-Group/blob/987e163a5b666d4b188e504dc9f37183625b6d50/Annual_income_gender.png)

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

![Age - Gender](https://github.com/Ktiscar1/Marketing-Group/blob/33a44d051155ccf3abf487d64e8da25ae6aabcf8/age_gender.png)

![Spending Score - Gender](https://github.com/Ktiscar1/Marketing-Group/blob/1357b24316666f0a79843ef0794b085f78e2925d/spending_score_gender.png)

# Bivariarte Analysis

A scatter plot is employed to assess the relationship between "Spending Score" and "Annual Income." Preliminary visual inspection, without the application of advanced analytical techniques, indicates the presence of distinct groupings within the data. Specifically, the observations suggest the existence of approximately five clusters, as well as a limited number of outliers.

![Scatter plot between Annual Income and Spending Score](https://github.com/Ktiscar1/Marketing-Group/blob/b26462ce4f251c86f458d59538b992b249110296/scatter_plot_income_spending.png)

To efficiently examine multivariate relationships, marginal distributions, and structural patterns within the dataset, a pair plot is employed, with gender used as a categorical differentiating variable. This visualization facilitates the preliminary identification of potential cluster structures in the data prior to formal clustering analysis.

![Pair Plot](https://github.com/Ktiscar1/Marketing-Group/blob/a8e791ecf5653cf74437b2ed186c43255c252a93/paitplot.png)

As a last step, a heatmap can be generated to observe the correlation between the data.

![Heatmap](https://github.com/Ktiscar1/Marketing-Group/blob/b86557ef7ce9d9f9cdb7ff57cf2033fc42fcbf74/heatmap.png)

# K-Means Algorithm Clustering (Univariate, Bivariate & Multivarite)

# Univariate Clustering

In the next step, the Annual Income data is fitted to a K-means clustering algorithm, which partitions the dataset into distinct groups. The resulting cluster labels are then compared with the original data by assigning them as a new feature:


```python
df['Income Cluster'] = clustering1.labels_
```
<img width="927" height="284" alt="image" src="https://github.com/user-attachments/assets/d8d9ccd4-963d-4491-80d5-670072b09b04" />

This approach enables the computation of summary statistics for each identified cluster.


*(This process is iterative and is modified with each pass to obtain optimal number of clusters)*
Now is possible to do summary statistics around the univariate cluster.

We can check how many of the customers fall on each cluster.
```python
df['Income Cluster'].value_counts()
```
<img width="262" height="255" alt="image" src="https://github.com/user-attachments/assets/016a66a8-7061-4283-8d89-3a7cc0651cd4" />

We can see that cluster #2 contains a higher count of customers and cluster #1 contains a lower count of customers.

```python
clustering1.inertia_
```
This represents the distance between the centroids.
*(Clusters from range 1 to 11 were generated to to obtained ideal clusters)*
*(This table contains optimal number of cluster, for whole proceess check Colab Notebook)*

The next step is to fit the possible clusters in "Annual Income" and append them to the inertia score.

```python
inertia_scores=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(df[['Annual Income (k$)']])
  inertia_scores.append(kmeans.inertia_)
```

<img width="282" height="248" alt="image" src="https://github.com/user-attachments/assets/cb39a306-2ceb-4f3a-a6a0-dcbb4a71d910" />

An elbow plot is generated to determine the optimal number of clusters. It is observable that the elbow occurs between clusters 2 and 4 meaning that the optimal number of clusters is 3.

![Elbow Plot](https://github.com/Ktiscar1/Marketing-Group/blob/e06bee0a6930fc5b661e74ce0f7954ffaef9ba1d/Elbow%20plot.png)

*(After conducting an analysis of the iterative process to obtain the optimal number of clusters is possible to produce the following table)*

By aggregating "Age, "Annual Income," and "Spending Score" to "Income Clusters" and obtaining the mean foe the tables is possible to observe that people in cluster 0 contains the highest annual incomme, the ones in cluster 2 contains the "middle" income and cluster 1 is the demographic with less annual income.

<img width="714" height="261" alt="image" src="https://github.com/user-attachments/assets/87618af0-bb18-4fb7-8301-73ec7dbd6e7b" />

# Bivariate Clustering

*(For this section can use same method to find the ideal number of clusters)*

The "Annual Income" and "Spending Score" columns need to be fitted to transform raw, unlabeled data into actionable, categorized information. This produce the following table:

<img width="1222" height="286" alt="image" src="https://github.com/user-attachments/assets/e69cd0b5-ebd8-404a-87fe-a55a1531c1b9" />

This table can be optimized using the optimal number of clusters (The same method used in the univariate clustering can be used here).

```python
inertia_scores2=[]
for i in range(1,11):
  kmeans2=KMeans(n_clusters=i)
  kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
  inertia_scores2.append(kmeans2.inertia_)
```

With this optimization it is possible to generate an elbow table and create a new scatter plott to gather relevant information regarding the spending trend between the demographic.

![Elbow Plot Bivariate](https://github.com/Ktiscar1/Marketing-Group/blob/a62d452690e83775bee730eb1a24fc3c9d9720d0/Elbow%20Plot%20Bivariate.png)

After analysing the elbow plot it is possible to observe that the elbow occurs between clusters 4 and 6 meaning that the optimal number of clusters is 5. 

With this new information the next step is to do a visual analysis with the correct number of clusters. A scatter plot allow the visualization of relationships, correlations and patter between the variables. 

![Scatter Plot](https://github.com/Ktiscar1/Marketing-Group/blob/4beb490062287707ec5e2d80d2a0513fbeb99320/scatter%20plot%20bivariate.png)

Using "Annual income" as the x-axis and the "Spending Score" as the y-axis it is possible to determining the following information:

* Cluster 0 contains high income and high spending.
  * "Premium customers could be located in this demographic.
* Cluster 1 contains low income and high spending.
  * Enthusiastic spenders despite lower income.
* Cluster 2 contains high income and low spending.
  * Financially capable but not engaged. This is untapped potential.
* Cluster 3 contains mid income and mid spending.
  * Reliable costumers.
* Cluster 4 contains very high income and mid spending (closer to low spending).
  * High earners with inconsistent behavior
* Clusters 0 and 1 are the strongest revenue drivers.
* cluster 2 is the biggest growth opportunity.
* Cluster 3 contains the demographic that spends as much as it earns.
* Cluster 4 may be worth deeper analysis..

It is possible to check for gender and average age of the clusters for deeper analysis
<img width="499" height="313" alt="image" src="https://github.com/user-attachments/assets/cd257023-935a-452b-8521-5b2bef160db2" />

<img width="850" height="349" alt="image" src="https://github.com/user-attachments/assets/ec90c17e-6eb8-497e-bb6c-c1d7aaa25090" />

# Multivariate Analysis

The first step for handling multivariate analysis is to do scale the data to allow the algorithm to ensure that features with larger numerical ranges do not disproportionately dominate. 

In order to scale the data correctly the following steps are follow

* Apply one hot encoder to "Gender" column to get rid of male and female labels and instead replacing it with 0 and 1
* Dropping the useless colums
    * "Gender" - Strings
    * "Income Cluster"
    * "Spending and Income Cluster"

<img width="693" height="284" alt="image" src="https://github.com/user-attachments/assets/89dbfe4e-2328-47a1-9a1c-162c3d860db5" />

Now it is possible to scale the data using the new table into a new dataframe to generete the correct analysis.

```python
inertia_scores3=[]
for i in range(1,11):
  kmeans3 = KMeans(n_clusters=i)
  kmeans3.fit(dff)
  inertia_scores3.append(kmeans3.inertia_)
```

![Elbow Multivariate](https://github.com/Ktiscar1/Marketing-Group/blob/1a2bfb452635ae54c2a154216330bdcaa1a2b567/Elbow%20Plot%20Multi.png)

After analysing the elbow plot it is possible to observe that the elbow occurs between clusters 3 and 6 where cluster 4 shows a curve that starts to flatten with each additional step.

*(More analysis can be done but is not needed to obtain thee relevant needed information)*

# Summarize Statics on the clusters

![Scatter Plot](https://github.com/Ktiscar1/Marketing-Group/blob/4beb490062287707ec5e2d80d2a0513fbeb99320/scatter%20plot%20bivariate.png)

* Target group would be cluster 0 which has a high "Spending Score"
* 54% of cluster 0 shoppers are women. Marketing team shoul look for ways to attract these costumers using marketing campaign targeting popular intems in this demographic
* Cluster 1 presents an opportunity to market to the customers for sales event in popular items.
