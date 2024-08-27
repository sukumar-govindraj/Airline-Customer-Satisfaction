
## Airline Customer Satisfaction Survey

The project involves analyzing airline satisfaction data to draw meaningful insights using various data science techniques.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Description](#data-description)
5. [Methodology](#methodology)
    - [Descriptive Analysis](#descriptive-analysis)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Feature Engineering](#feature-engineering)
    - [Model Building and Evaluation](#model-building-and-evaluation)
6. [Results](#results)
    - [Key Findings](#key-findings)
    - [Model Performance](#model-performance)
7. [Conclusion](#conclusion)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

This project aims to analyze airline satisfaction data using Python to understand factors influencing customer satisfaction. The project employs various data analysis and machine learning techniques to explore the dataset, identify patterns, and build predictive models.

## Installation

To replicate this analysis, you need to have Python installed on your system. The necessary Python packages can be installed using:

```bash
pip install -r requirements.txt
```

## Usage

To run the analysis, open the Jupyter notebook and run the cells sequentially. The notebook is structured to guide you through the process, from data loading to the final analysis.

## Data Description

The dataset used for this project contains various features related to airline satisfaction. The key variables include customer feedback on different aspects of the airline service, demographic information, and satisfaction levels.

## Methodology

### Descriptive Analysis

The initial part of the analysis involves summarizing the data to get an understanding of the distribution and central tendencies of the variables. Summary statistics and initial observations are provided.

### Exploratory Data Analysis (EDA)

The Exploratory Data Analysis (EDA) section is crucial for understanding the structure and patterns within the dataset. This analysis includes both descriptive statistics and visualizations to uncover relationships between variables.

#### Principal Component Analysis (PCA)

Principal Component Analysis (PCA) was used to reduce the dimensionality of the dataset while retaining the most significant features. The steps involved in the PCA are as follows:

1. **Feature Standardization**: All numerical features were standardized to ensure they contribute equally to the analysis.
2. **PCA Application**: PCA was applied to the standardized features to identify the principal components.
3. **Feature Loadings**: The loadings of each feature for the first principal component were calculated and ranked to determine the most influential features.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df_airline['Gender'] = df_airline['Gender'].replace({'Male': 0, 'Female': 1})
X = df_airline.drop('satisfaction', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
loadings = pca.components_[0]
feature_loadings = pd.DataFrame({'Feature': X.columns, 'Loading': loadings})
feature_loadings['Loading_abs'] = feature_loadings['Loading'].abs()
feature_loadings = feature_loadings.sort_values(by='Loading_abs', ascending=False)
```

This analysis identified key features that were retained for further modeling, improving the efficiency and accuracy of the models.

### Feature Engineering

Features are created and transformed based on the insights from EDA to improve the performance of predictive models. This includes handling missing data, encoding categorical variables, and scaling numeric features.

### Model Building and Evaluation

Several machine learning models are built to predict customer satisfaction. The models include:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**

Each model is evaluated using the following metrics:

- **Accuracy**: The proportion of correctly classified instances.
- **Precision, Recall, F1-Score**: To evaluate the performance of models in distinguishing between satisfied and dissatisfied customers.
- **Confusion Matrix**: To visualize the performance of the classification models.
- **ROC Curve & AUC Score**: To measure the ability of the model to distinguish between classes.

## Results

### Key Findings

- **Correlation Insights**: Discusses key relationships identified between variables, such as the strong correlation between customer service ratings and overall satisfaction.
- **Important Features**: Lists the top features impacting customer satisfaction as identified by the models.

### Model Performance

- **Logistic Regression**: Provides a summary of the accuracy, precision, recall, and F1-score for the logistic regression model.
- **Random Forest**: Discusses the feature importance and performance metrics for the Random Forest model.
- **XGBoost**: Highlights the superior performance of XGBoost in terms of accuracy and AUC score.

## Conclusion

This section summarizes the insights gained from the analysis, including the importance of certain features in predicting customer satisfaction. The results suggest actionable insights for improving airline services.


- **Overall Customer Satisfaction Distribution**:

The dataset contains a significant number of customers who are either "satisfied" or "neutral/dissatisfied," indicating a diverse range of customer experiences.


- **Customer Satisfaction by Travel Class**:

Customers in the Business class tend to report higher satisfaction levels compared to those in Eco and Eco Plus classes. This suggests that premium services are more likely to meet or exceed customer expectations.

- **Customer Satisfaction by Customer Type**:

Loyal customers exhibit a much higher satisfaction rate than disloyal customers. This highlights the importance of retaining loyal customers by continually meeting their expectations.

-**Customer Satisfaction by Gender**:

The satisfaction levels between male and female customers are relatively similar, indicating no significant gender-based differences in overall satisfaction.

-**Customer Satisfaction by Age Group**:

Younger customers (19-35) seem to report higher satisfaction levels, while older age groups (51-65) show a relatively balanced distribution between satisfaction and dissatisfaction. This suggests that younger travelers might have different expectations or experiences compared

-**Correlation Matrix**:

The correlation matrix shows how different features relate to customer satisfaction and each other. Notably:
Inflight entertainment, Online boarding, and Seat comfort have relatively higher positive correlations with satisfaction.
Departure and arrival delays are negatively correlated with satisfaction, though the correlations are not very strong.


-**Impact of Flight Distance on Satisfaction:**

There is a noticeable trend where customers on shorter flights tend to have slightly lower satisfaction levels compared to those on longer flights. This could indicate that passengers on longer flights might appreciate the services more, possibly due to greater expectations and experiences over extended periods.

-**Service Ratings vs. Satisfaction**:

Service features like Inflight entertainment, Seat comfort, On-board service, and Cleanliness show a strong association with higher satisfaction ratings. Customers who rate these services highly are more likely to be satisfied.
Conversely, low ratings in these areas are associated with dissatisfaction, suggesting these are critical areas for maintaining high customer satisfaction.

-**Impact of Departure and Arrival Delays on Satisfaction**:

Both departure and arrival delays are associated with lower satisfaction. Passengers experiencing significant delays tend to report dissatisfaction, highlighting the importance of punctuality in the overall customer experience.

## Business Report Conclusions:

- Enhancing Premium Services: Focus on maintaining and enhancing services for Business class travelers, as they are the most satisfied segment.
- Loyalty Programs: Continue investing in loyalty programs as they significantly contribute to customer satisfaction.
- Targeted Improvements: Consider targeted improvements for Eco and Eco Plus classes to elevate their satisfaction levels, particularly for older travelers who may have distinct needs.
- Service Enhancement Focus: The strong correlation between inflight entertainment, seat comfort, and satisfaction indicates these areas are critical to customer happiness. Improving these services could lead to higher overall satisfaction.
- Flight Experience: Longer flights tend to have higher satisfaction, potentially due to more opportunities for service engagement.Airlines could focus on maintaining and enhancing service quality, especially on shorter flights where satisfaction dips slightly.
- Punctuality: Delays are a key dissatisfier. Efforts to reduce delays or better manage customer expectations during delays could significantly impact satisfaction scores.
