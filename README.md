# Peak-ProphetPro
## Inspiration
Peak-ProphetPro was inspired by the Chevron challenge, focusing on predicting peak oil production across various well profiles.

## What it does
Peak-ProphetPro has not only a Neural Network for predicting peak oil production, but also a fullstack data visualization dashboard using Taipy! This dashboard allows users to examine the dataset and generate dynamic graphs, including heatmaps, scatter plots, and histograms.


![ScreenRecording2024-01-20at3 53 55PM-ezgif com-video-to-gif-converter](https://github.com/alisonqiu/2024datathon/assets/90943803/12123ff8-0636-430b-a0bb-1d7a5a3643e9)


## How we built it
### Handling Missing Values:
We addressed missing values by identifying rows without missing target values and assessing the proportion of missing values in specific variables. For instance, we dropped the 'frac_type' variable because all rows with non-missing target values had the same 'frac_type,' making it irrelevant for the model's learning process.

### Imputing Numerical Values:
To decide between using the mean or median for imputing numerical values, we examined data distributions. Since the data distribution was skewed or contained outliers, we opted for the median, which is more suitable when dealing with asymmetric distributions or outliers.

### Imputing Categorical Variables:
We decided against the common way of replacing missing values in categorical variables because there is no significant differences between the top few frequencies. We tried random sampling, and we later improved our performance by building a predictive model to impute missing values.

### Modeling:
We built four baseline models (Linear Regression, Decision Trees, Random Forest, XGBoost) and an ensemble of Random Forest and XGBoost. We also built a Neural Network with four hidden layers. The final model with best performance is a Stacking Ensemble. The ensemble consists of three base models: RandomForestRegressor, XGBRegressor and LinearRegression. These models are combined using a StackingRegressor with a final estimator being another Linear Regression model.
<img width="1050" alt="Screenshot 2024-01-20 at 8 02 49 PM" src="https://github.com/alisonqiu/2024datathon/assets/90943803/2af72c2e-2a41-4ef7-9d2e-5336cbbf4424">

## What's next for Peak-ProphetPro
Although we wrote the code for model training prediction, we didn't have time to add it as a webpage,  so we will integrate the model training prediction and management functionality into the taipy visualization tool! 

