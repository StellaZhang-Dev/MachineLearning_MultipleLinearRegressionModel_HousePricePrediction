# MachineLearning_MultipleLinearRegressionModel_HousePricePrediction

## Project Overview
This project aims to predict **house prices** in Taiwan using a **Multiple Linear Regression Model**. The dataset contains various attributes related to real estate, including geographic, economic, and environmental factors. The primary goal of this project is to build a predictive model that estimates house prices based on these features and to evaluate the model’s performance in terms of accuracy and significance of the variables.

## Dataset
The dataset used for this project is **`real-estate-taiwan.csv`**, which includes several features related to the housing market in Taiwan.

### Features:
- **Transaction Date**: The transaction date for the house (year-month format).
- **House Age**: The age of the house (in years).
- **Distance to Nearest MRT Station**: The distance to the nearest MRT (subway) station (in meters).
- **Number of Convenience Stores**: The number of convenience stores near the house.
- **Latitude**: The latitude coordinates of the house.
- **Longitude**: The longitude coordinates of the house.
- **Price Per Unit Area**: The target variable, representing the price per square meter (NTD).

### Data Preprocessing
1. **Handling Missing Values**: Missing values were handled by removing the rows containing them.
2. **Feature Scaling**: Features like `Distance to Nearest MRT Station` and `House Age` were standardized using **min-max scaling**.
3. **Train-Test Split**: The dataset was split into a training set (80%) and a testing set (20%) to evaluate the model’s performance.

## Modeling Approach

### Multiple Linear Regression Model
For this project, we utilized **Multiple Linear Regression** to predict house prices based on several input features. Multiple linear regression is a model that predicts a target variable as a linear combination of multiple predictor variables.

#### Model Equation:
The equation for the multiple linear regression model can be written as:
$$
\text{Price Per Unit Area} = \beta_0 + \beta_1(\text{House Age}) + \beta_2(\text{Distance to MRT}) + \dots + \epsilon
$$

Where:
   - \( \beta_0 \) is the intercept.
   - \( \beta_1, \beta_2, \dots \) are the coefficients for each predictor variable.
   - \( \epsilon \) is the error term.

### Key Steps:
1. **Model Training**:
   - The model was trained using the **training set** to learn the relationship between the features (e.g., house age, distance to MRT) and the target variable (price per unit area).
   - The coefficients were estimated using **Ordinary Least Squares (OLS)**, minimizing the sum of squared differences between the observed and predicted values.

2. **Model Evaluation**:
   - The model’s performance was evaluated on the **test set** using metrics such as:
     - **R-squared (R²)**: The proportion of variance in the dependent variable that is predictable from the independent variables.
     - **Adjusted R-squared**: Adjusted for the number of predictors in the model.
     - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
   - **P-values**: To evaluate the statistical significance of each predictor variable in the model.

### Code Structure
The core logic of the project is implemented in the **`Analysis_of_House_Prices.Rmd`** file. The R script is structured as follows:

1. **Data Loading**:
   - The dataset `real-estate-taiwan.csv` is loaded using the `read.csv()` function:
   ```r
   df <- read.csv("real-estate-taiwan.csv")
   ```

2. **Data Preprocessing**:

   - Missing values were handled, and feature scaling was applied to the dataset.

3. **Modeling**:

   - The `lm()` function in R was used to fit the multiple linear regression model:
   ```r
   model <- lm(PricePerUnitArea ~ HouseAge + DistanceToMRT + NumberOfStores + Latitude + Longitude, data = train_data)
   ```
   
4. **Model Evaluation**:

   - The model’s predictions on the test set were evaluated using a variety of performance metrics, such as `R-squared`, `Adjusted R-squared`, and `MSE`.

5. **Visualization**:

   - A scatter plot was created to visualize the relationship between **MRT distance** and **house price**:
   ```r
   ggplot(data = house_data, aes(x = mrt_distance, y = house_price)) +
     geom_point() +
     labs(title = "House Price vs Distance to MRT")
     ```
     
## Model Performance
   - **R-squared (Training Set)**: The model achieved an **R-squared value of 0.5712**, indicating that **57.12**% of the variance in house prices can be explained by the predictor variables.
   - **Adjusted R-squared**: The adjusted R-squared value of **0.5659**, accounting for the number of predictors in the model.
   - **Mean Squared Error (Test Set)**: The mean squared error was **8.965**, showing the average squared difference between predicted and actual house prices.

### Key Insights from the Model:
   - **House Age**: Older houses tend to have lower prices per unit area, which is consistent with real estate market trends.
   - **Distance to MRT Station**: Houses located closer to MRT stations generally have higher prices, reflecting the premium for better accessibility.
   - **Number of Convenience Stores**: The availability of more convenience stores positively correlates with house prices.

## Results
   - The model effectively predicts house prices based on the features provided. Statistically significant variables include **house age**, **distance to MRT station**, **number of convenience stores**, and **latitude**.

## Conclusion
This project successfully applied a **Multiple Linear Regression Model** to predict house prices in Taiwan based on a variety of features. The results highlight the influence of location-related variables such as proximity to MRT stations and the number of convenience stores on house prices. The model can serve as a useful tool for understanding the real estate market in Taiwan.

## Future Work
   - **Feature Engineering**: Explore additional variables, such as neighborhood quality or housing trends over time, to enhance the model.
   - **Non-Linear Models**: Test non-linear regression models or **decision tree-based models** like **Random Forest** or **XGBoost** to capture potential non-linear relationships in the data.
   - **Cross-Validation**: Implement cross-validation to ensure the model’s robustness and generalizability.

## Technologies Used
   - **R Programming Language**: For data manipulation, modeling, and visualization.
   - **ggplot2**: For creating visualizations.
   - **lm()**: For fitting multiple linear regression models.
   - **dplyr**: For data preprocessing and manipulation.

## How to Run
1. Clone the repository:
   ```r
   git clone git@github.com:StellaZhang-Dev/MachineLearning_MultipleLinearRegressionModel_HousePricePrediction.git
   ```

2. Install required packages in R:
   ```r
   install.packages(c("ggplot2", "dplyr", "corrplot"))
   ```

3. Run the R script:
   ```r
   source("Analysis_of_House_Prices.Rmd")
   ```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

