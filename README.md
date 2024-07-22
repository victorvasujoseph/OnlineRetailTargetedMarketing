# Online Retail Customer Value Classification

This repository contains a machine learning project for classifying customers into high-value and low-value categories based on their purchasing behavior. The model is built using a K-Nearest Neighbors (KNN) classifier and includes data preprocessing, model training, evaluation, and visualization.

## Overview

The goal of this project is to predict whether a customer is high-value or low-value based on their purchasing history. High-value customers are identified as those whose total spending is in the top 25%.

### Key Features

- Data Preprocessing: Aggregation of customer data, calculation of total spending, and creation of target variable.
- Model Training: Using a pipeline to standardize features and train a KNN classifier.
- Hyperparameter Tuning: Using GridSearchCV to find the best number of neighbors for the KNN classifier.
- Model Evaluation: Calculating accuracy, F1 score, precision, recall, and plotting confusion matrix.
- Visualization: Plotting distribution of total spending and F1 score for different K values.

## Setup Instructions

### Prerequisites

- Python 3.x
- Jupyter Notebook (optional, but recommended for interactive exploration)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/online-retail-customer-value.git
    cd online-retail-customer-value
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Ensure that the dataset (`Online Retail.xlsx`) is in the root directory of the project.
2. Run the Jupyter Notebook or Python script:
    ```bash
    jupyter notebook online_retail_customer_value.ipynb
    ```
    OR
    ```bash
    python online_retail_customer_value.py
    ```

## Code Explanation

The main steps in the code are:

1. **Load the Dataset:**
    ```python
    data = pd.read_excel('OnlineRetail.xlsx')
    ```

2. **Data Preprocessing:**
    ```python
    customer_data = data.groupby('CustomerID').agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean',
        'InvoiceNo': 'nunique'
    }).reset_index()
    customer_data['TotalSpending'] = customer_data['Quantity'] * customer_data['UnitPrice']
    threshold = customer_data['TotalSpending'].quantile(0.75)
    customer_data['CustomerValue'] = (customer_data['TotalSpending'] > threshold).astype(int)
    ```

3. **Train-Test Split:**
    ```python
    X = customer_data[['Quantity', 'UnitPrice', 'InvoiceNo', 'TotalSpending']]
    y = customer_data['CustomerValue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ```

4. **Pipeline and Grid Search:**
    ```python
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {'knn__n_neighbors': range(1, 11)}
    f1_scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(pipeline, param_grid, scoring=f1_scorer, cv=5)
    grid_search.fit(X_train, y_train)
    ```

5. **Model Evaluation:**
    ```python
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    ```

6. **Visualization:**
    ```python
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.hist(customer_data['TotalSpending'], bins=50, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='High Value Threshold')
    plt.title('Distribution of Total Spending')
    plt.xlabel('Total Spending')
    plt.ylabel('Frequency')
    plt.legend()

    f1_scores = grid_search.cv_results_['mean_test_score']
    k_values = range(1, 11)
    plt.subplot(1, 3, 2)
    plt.plot(k_values, f1_scores, marker='o', color='blue')
    plt.title('F1 Score for Different K Values')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('F1 Score')
    plt.xticks(k_values)

    plt.subplot(1, 3, 3)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Low Value', 'High Value'], yticklabels=['Low Value', 'High Value'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()
    ```

## Conclusion

**Best Parameters:**
- Number of Neighbors (K): 3

**Performance Metrics on Test Set:**
- Accuracy: 0.961 (96.1%)
- F1 Score: 0.920 (92.0%)
- Precision: 0.910 (91.0%)
- Recall: 0.931 (93.1%)

The model performs well with high accuracy and balanced precision and recall, making it effective for identifying high-value customers. This can be useful for targeting personalized marketing strategies and improving customer segmentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
