# Instacart Dashboard

This folder contains a simple static dashboard for the Data Mining project.

## What it shows

- Dataset KPIs: orders, customers, products, departments, aisles, order lines
- Basket KPIs: average basket size, multi-item baskets, reorder rate
- FP-Growth KPIs: association rule count, average confidence, maximum lift
- Visual summaries for top products, departments, reorder-heavy departments, order timing, and strongest association rules
- Detailed table of the top generated rules

## How to generate

From the project root:

```powershell
python dashboard/generate_dashboard.py
```

Then open:

```text
dashboard/index.html
```

The dashboard uses only `pandas` and plain HTML/CSS, so it does not need Dash, Streamlit, Plotly, or an internet connection.
