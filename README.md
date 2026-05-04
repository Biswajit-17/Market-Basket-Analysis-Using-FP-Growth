# Instacart Market Basket Analysis (FP-Growth)

## Project Overview
This project performs Market Basket Analysis on the Instacart grocery dataset to discover hidden patterns in customer purchasing behavior. It uses the **FP-Growth (Frequent Pattern Growth)** algorithm to find products that are frequently bought together and generates actionable association rules. 

We specifically chose FP-Growth over Apriori because of its superior performance and memory efficiency when handling massive, real-world datasets like Instacart's millions of orders.

## How to Run

1. Ensure the Instacart dataset files (`order_products__prior.csv` and `products.csv`) are located in an `Instacart/` folder within the same directory as the script.
2. Install the required dependencies using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the Python script:
   ```bash
   python fpgrowth_analysis.py
   ```
   *(Note: This will output progress to the console, generate a `fpgrowth_association_rules.csv` file, and pop up two interactive DataViz charts on your screen).*

## Current Implementation Pipeline

The pipeline is entirely contained within `fpgrowth_analysis.py` and executes in 6 distinct steps:

### Step 1: Data Loading & Joining
* **What it does:** Reads the `order_products__prior.csv` (containing past order contents) and `products.csv` (containing product names). It joins these tables so we can work with human-readable product names instead of numeric IDs.
* **Optimization:** Drops unneeded columns early to save memory, keeping only `order_id` and `product_name`.

### Step 2: Basket Grouping & Preprocessing
* **What it does:** Groups all items belonging to the same `order_id` into a single list (a "basket").
* **Optimization (Noise Reduction):** Automatically filters out and drops any orders containing only 1 item. Single-item baskets cannot form associations and only dilute the metrics.

### Step 3: Sparse Matrix Encoding
* **What it does:** Converts the variable-length lists of products into a massive boolean matrix (One-Hot Encoding). Each row is an order, and each column is a product (True if bought, False if not).
* **Optimization:** Uses `TransactionEncoder(sparse=True)` to create a SciPy sparse matrix. Instead of allocating RAM for millions of "False" zeroes, it only stores the "True" values. This prevents the system from crashing with Out-of-Memory (OOM) errors.

### Step 4: FP-Growth Execution
* **What it does:** Scans the sparse matrix to find "Frequent Itemsets" (product combinations bought together).
* **Threshold:** Uses initial `min_support=0.001` (0.1% of multi-item orders) to capture a wide base of patterns before stricter filtering.

### Step 5: Association Rule Generation & Scoring
* **What it does:** Converts the frequent itemsets into "If A, then B" directional rules.
* **Strict Filtering:** 
  * `support >= 0.004`: Requires the rule to appear in at least 0.4% of orders.
  * `confidence >= 0.15`: Requires at least a 15% chance they will buy item B given item A.
  * `lift >= 1.5` and `lift <= 10`: Ensures items are meaningfully linked, but explicitly caps lift at 10 to filter out hyper-niche, duplicate-style variants (like flavors of the same obscure yogurt).
* **Custom Metric (`business_score`):** Calculates `Support * Confidence * Lift` to prioritize rules that are highly reliable, impactful, and broadly applicable.

### Step 6: Formatting & Export
* **What it does:** Converts native Python `frozenset` objects to clean, readable comma-separated strings before exporting to `fpgrowth_association_rules.csv`.

### Step 7: Visualization
* **What it does:** Uses `seaborn` and `matplotlib` to generate two bar charts instantly: top 10 rules by `business_score` and top 10 rules by `confidence`. 
* **Optimization:** Runs using `Qt5Agg` backend so plots appear directly on screen without needing to save image files to the computer.

## Dependencies
* `pandas`
* `numpy`
* `scipy`
* `mlxtend` 
* `matplotlib`
* `seaborn`
* `PyQt5`