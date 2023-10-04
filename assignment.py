import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def least_squares(y_real, y_ideal):
    return np.sum((y_real - y_ideal)**2)

def find_best_functions(y_real, ideal_functions):
    best_functions = {}
    for col in ideal_functions.columns:
        best_functions[col] = least_squares(y_real, ideal_functions[col])
    best_functions = sorted(best_functions.items(), key=lambda x: x[1])[:4]
    return [func[0] for func in best_functions]

def assign_test_data(test_data, training_data, ideal_functions, best_functions):
    mapping = {}
    for best_func in best_functions:
        largest_deviation = np.max(np.abs(training_data - ideal_functions[best_func]))
        threshold = largest_deviation * np.sqrt(2)
        
        for i, y_val in enumerate(test_data['y']):
            deviation = np.abs(y_val - ideal_functions.loc[i, best_func])
            if deviation <= threshold:
                mapping[(test_data.loc[i, 'x'], y_val)] = {'function': best_func, 'deviation': deviation}
    return mapping

# Load data
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
ideal_functions = pd.read_csv('ideal.csv')

# Create a dictionary to store mappings for each y-column in the training data
overall_mapping = {}

# Find best functions and assign test data for each y-column
for y_col in ['y1', 'y2', 'y3', 'y4']:
    best_functions = find_best_functions(training_data[y_col], ideal_functions)
    print(f"Best functions for {y_col} are: {best_functions}")
    
    mapping = assign_test_data(test_data, training_data[y_col], ideal_functions, best_functions)
    overall_mapping[y_col] = mapping

# Visualization (example for y1)
for func in find_best_functions(training_data['y1'], ideal_functions):
    plt.plot(training_data['x'], ideal_functions[func], label=f"{func} (Ideal for y1)")

plt.scatter(training_data['x'], training_data['y1'], c='r', label='Training Data (y1)')
plt.legend()
plt.show()

for func in find_best_functions(training_data['y2'], ideal_functions):
    plt.plot(training_data['x'], ideal_functions[func], label=f"{func} (Ideal for y2)")

plt.scatter(training_data['x'], training_data['y2'], c='r', label='Training Data (y2)')
plt.legend()
plt.show()

for func in find_best_functions(training_data['y3'], ideal_functions):
    plt.plot(training_data['x'], ideal_functions[func], label=f"{func} (Ideal for y3)")

plt.scatter(training_data['x'], training_data['y3'], c='r', label='Training Data (y3)')
plt.legend()
plt.show()

for func in find_best_functions(training_data['y4'], ideal_functions):
    plt.plot(training_data['x'], ideal_functions[func], label=f"{func} (Ideal for y4)")

plt.scatter(training_data['x'], training_data['y4'], c='r', label='Training Data (y4)')
plt.legend()
plt.show()