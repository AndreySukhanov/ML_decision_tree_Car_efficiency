# Car Efficiency Prediction with Decision Trees

## Description
This project classifies cars as "Economical" or "Gas Guzzler" based on fuel consumption (liters per 100 km) using a Decision Tree algorithm from Scikit-learn. It’s a fun and practical example of machine learning classification, focusing on automotive efficiency!

## Data
- **Source**: UCI Machine Learning Repository (Auto MPG dataset)
- **Features**: 
  - Horsepower
  - Weight
  - Cylinders
- **Target**: Efficiency (0 = Economical if < 10 L/100 km, 1 = Gas Guzzler if >= 10 L/100 km)

## Results
- Accuracy on test data: ~80–90%
- Confusion Matrix shows how well the model distinguishes between economical and gas-guzzling cars

## Visualizations
- **Decision Tree**: Displays the decision-making process of the model
- **Efficiency Scatter Plot**: Visualizes fuel consumption vs horsepower with color coding for efficiency

## Requirements
- Python 3.x
- Libraries: numpy, pandas, matplotlib, scikit-learn

## How to Run
1. Install dependencies: `pip install numpy pandas matplotlib scikit-learn`
2. Run the script: `python car_efficiency_tree.py`

## Visualizations
- [decision_tree.png](#)
- [efficiency_scatter.png](#)

## Notes
This project uses a synthetic threshold (10 L/100 km) to define efficiency. For real-world applications, you can adjust the threshold or add more features like Acceleration or Model Year.
