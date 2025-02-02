# Simple Linear Regression

## Description
This project involves coding a simple linear regression algorithm from scratch using NumPy. The gradient descent algorithm and cost function are also implemented manually. Additionally, the project includes functionality for loading datasets, splitting data into training and testing sets, and evaluating model performance using R² score and Mean Squared Error (MSE).

## Usage
1. Run the `linear_regression.py` file.
2. The script will either generate synthetic data or load a dataset, fit a linear regression model using gradient descent, and plot the results.
3. Performance metrics such as R² score and Mean Squared Error will be displayed in the console.

## Techniques
- Linear Algebra (matrices, vectors)
- Calculus (derivatives)
- NumPy usage
- Data splitting (training and testing sets)
- Performance evaluation (R² score, MSE)

## Files and Directories
- `linear_regression.py`: The main Python script that contains the linear regression implementation.
- `data/`: Directory to store dataset files (e.g., `dataset.csv`).
- `plots/`: Directory where generated plots will be saved.

## Example Analyses
### Synthetic Data
The script generates synthetic data and fits a linear regression model to it. The following plots are generated:
- **Training Data Fit**: The regression line fitted to the training data.
- **Test Data Fit**: The regression line fitted to the test data.
- **Cost Function History**: The change in the cost function value over iterations of gradient descent.

### Performance Metrics
The script calculates and displays the following performance metrics:
- **R² Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.

## Running the Script
1. Ensure you have the required dependencies installed:
   ```bash
   pip install numpy matplotlib scikit-learn
2. Run the script:
   ```bash
   python linear_regression.py
3. If using a custom dataset, place the dataset file in the data/ directory and update the file_path variable in the script.

## Project Structure
simple-linear-regression/
├── data/
│   └── dataset.csv (optional)
├── plots/
│   ├── linear_regression_fit_training.png
│   ├── linear_regression_fit_test.png
│   └── cost_history.png
├── linear_regression.py
└── README.md

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn

## Example
- Console Output
- R² Score: 0.65
- Mean Squared Error: 0.92

Generated Plots
- linear_regression_fit_training.png: Shows the regression line fitted to the training data.
- linear_regression_fit_test.png: Shows the regression line fitted to the test data.
- cost_history.png: Shows the cost function history during gradient descent.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
