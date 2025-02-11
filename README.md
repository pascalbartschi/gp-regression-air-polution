# Gaussian Process Regression for Air Quality Prediction

This project models air pollution (\(PM_{2.5}\)) concentrations using Gaussian Process Regression (GPR). The goal is to predict pollution levels at unmeasured locations using spatial data, while minimizing underpredictions in residential zones using an asymmetric loss function.

## Key Features
- **Kernel Selection**: Utilized a composite kernel: 
  \[
  k(x, x') = \sigma_0^2 + \text{const} \cdot Matern(\nu=1.5) + \text{WhiteNoise}
  \]
  for effective modeling of spatial correlations and measurement noise.
- **Normalization**: Applied Z-score normalization for the target values:
  \[
  \hat{y} = \frac{y - \mu}{\sigma}
  \]
- **Data Reduction**: Used KMeans clustering for undersampling, reducing the computational overhead of GPR inference.
- **Custom Cost Function**: Incorporated an asymmetric loss function:
  \[
  \ell_w(f(x), \hat{f}(x)) = (f(x) - \hat{f}(x))^2 \cdot 
  \begin{cases} 
    50 & \text{if } \hat{f}(x) \leq f(x) \text{ and area\_id } = 1 \\ 
    1 & \text{otherwise} 
  \end{cases}
  \]

## Implementation
### Main Functions
1. `train_model`: Fits the GPR model to normalized training data.
2. `generate_predictions`: Generates predictions and incorporates uncertainty for residential areas.
3. `extract_area_information`: Preprocesses data to extract spatial and area-specific information.

### Cost Function
The overall cost is defined as:
\[
L(\hat{f}) = \frac{1}{n} \sum_{i=1}^{n} \ell_w(f(x_i), \hat{f}(x_i))
\]

## Results
- Achieved accurate pollution predictions with minimal underestimation in residential zones.
- Visualization tools demonstrate predictions across the spatial grid.
- **Final asymmetric MSE cost:** 5.112 (ranked 72/296)
