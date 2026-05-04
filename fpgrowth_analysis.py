'''
Instacart Online Grocery Basket Analysis using FP-Growth Algorithm

Dataset Link: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset
Github Repository: https://github.com/Biswajit-17/Market-Basket-Analysis-Using-FP-Growth 
-- Dependencies --
pandas
numpy
scipy
mlxtend
matplotlib
seaborn
PyQt5
'''

import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # Bulletproof interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Step 1: Data Loading & Joining
print("Loading data...")
orders = pd.read_csv('Instacart/order_products__prior.csv')
products = pd.read_csv('Instacart/products.csv')

print("Merging data...")
merged_data = pd.merge(orders, products, on='product_id')
df = merged_data

print("Merged Data")
print(df.head())

# Keep only the columns we need for the FP-Growth algorithm
df = merged_data [['order_id', 'product_name']]



# Step 2: Basket Grouping/Aggregation
print("\nGrouping items into baskets...")
# Group by order_id and aggregate the product names into a list for each order
baskets = df.groupby('order_id')['product_name'].apply(list)

# Drop single-item baskets to remove noise (they can't form association rules anyway)
baskets = baskets[baskets.apply(len) > 1]

print(f"Total number of multi-item orders (baskets): {len(baskets)}")

# Print a few baskets to verify
print("Sample baskets:")
print(baskets.head())



# Step 3: Transaction Encoding (One-Hot Encoding)
print("\nEncoding baskets into a sparse boolean matrix...")
te = TransactionEncoder()

# We use sparse=True to return a SciPy sparse matrix instead of a standard NumPy array.
# This prevents our computer from immediately running out of memory (RAM).
te_ary = te.fit(baskets).transform(baskets, sparse=True)

# Convert the sparse matrix back into a memory-efficient Pandas DataFrame
# The columns are the unique product names we tracked.
df_encoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

print("Matrix encoding complete.")
print(f"Matrix shape: {df_encoded.shape} (Rows: Orders, Cols: Products)")



# Step 4: FP-Growth Execution
print("\nRunning FP-Growth to find frequent itemsets (min_support=0.001)...")
# min_support=0.001 means the item(s) must appear in at least 0.1% of multi-item orders
frequent_itemsets = fpgrowth(df_encoded, min_support=0.001, use_colnames=True)

print("Frequent itemsets generation complete.")
print(f"Total frequent itemsets found: {len(frequent_itemsets)}")
print(frequent_itemsets.sort_values('support', ascending=False).head(10).assign(itemsets=lambda df: df.itemsets.apply(lambda x: ', '.join(x))))



# Step 5: Association Rule Generation
print("\nGenerating Association Rules (metric='lift', min_threshold=1.5)...")
# Calculate rules using the frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

# Filter rules to keep useful and explainable results
rules = rules[
    (rules["support"] >= 0.004) &
    (rules["confidence"] >= 0.15) &
    (rules["lift"] >= 1.5) &
    (rules["lift"] <= 10)
]

# Create the business score and sort by it
rules['business_score'] = rules['support'] * rules['confidence'] * rules['lift']
rules = rules.sort_values('business_score', ascending=False)

print("Association rule generation complete.")
print(f"Total rules discovered: {len(rules)}")
print("Top 10 strongest associations (by business score):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'business_score']].head(10))



# Step 6: Export Results
print("\nExporting rules to CSV...")

# 'antecedents' and 'consequents' are returned as frozenset objects by default.
# Converting them to clean, readable strings before saving to CSV.
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

output_filename = 'fpgrowth_association_rules.csv'
# Keep only the specified columns for the final export
export_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'business_score']
rules[export_columns].to_csv(output_filename, index=False)

print(f"Results successfully saved to '{output_filename}'.")


# Step 7: Generate simple graphs
print("\nGenerating simple graphs...")

sns.set_theme(style="whitegrid")
# Make a readable rule label for graphs
rules["rule"] = rules["antecedents"] + " -> " + rules["consequents"]
rules["rule"] = rules["rule"].apply(lambda x: x if len(x) <= 55 else x[:52] + "...")

# Graph 1: Top rules by business score
top_business_score = rules.sort_values("business_score", ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_business_score, x="business_score", y="rule", color="gray")
plt.title("Top 10 Association Rules by Business Score")
plt.xlabel("Business Score")
plt.ylabel("Association Rule")
plt.tight_layout()

# Graph 2: Top rules by confidence
top_confidence = rules.sort_values("confidence", ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_confidence, x="confidence", y="rule", color="gray")
plt.title("Top 10 Association Rules by Confidence")
plt.xlabel("Confidence")
plt.ylabel("Association Rule")
plt.tight_layout()
plt.show()

print("Project complete! CSV was generated and plots were displayed successfully.")
