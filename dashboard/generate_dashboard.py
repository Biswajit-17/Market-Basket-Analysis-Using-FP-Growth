from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Instacart"
RULES_FILE = ROOT_DIR / "fpgrowth_association_rules.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "index.html"

DOW_LABELS = {
    0: "Sun",
    1: "Mon",
    2: "Tue",
    3: "Wed",
    4: "Thu",
    5: "Fri",
    6: "Sat",
}


def format_number(value: float | int) -> str:
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def add_series(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right
    return left.add(right, fill_value=0)


def order_product_files() -> Iterable[Path]:
    for filename in ("order_products__prior.csv", "order_products__train.csv"):
        path = DATA_DIR / filename
        if path.exists():
            yield path


def summarize_order_products() -> dict[str, object]:
    product_counts: pd.Series | None = None
    product_reorders: pd.Series | None = None
    basket_sizes: pd.Series | None = None
    total_rows = 0
    total_reorders = 0

    for path in order_product_files():
        chunks = pd.read_csv(
            path,
            usecols=["order_id", "product_id", "reordered"],
            chunksize=1_000_000,
        )
        for chunk in chunks:
            total_rows += len(chunk)
            total_reorders += int(chunk["reordered"].sum())

            product_counts = add_series(product_counts, chunk.groupby("product_id").size())
            product_reorders = add_series(
                product_reorders, chunk.groupby("product_id")["reordered"].sum()
            )
            basket_sizes = add_series(basket_sizes, chunk.groupby("order_id").size())

    if product_counts is None or product_reorders is None or basket_sizes is None:
        raise FileNotFoundError("No Instacart order product CSV files were found.")

    return {
        "product_counts": product_counts.astype("int64"),
        "product_reorders": product_reorders.astype("int64"),
        "basket_sizes": basket_sizes.astype("int64"),
        "total_rows": total_rows,
        "total_reorders": total_reorders,
    }


def bar_table(
    rows: list[dict[str, object]],
    label_key: str,
    value_key: str,
    value_formatter=format_number,
) -> str:
    if not rows:
        return "<p class=\"empty\">No data available.</p>"

    max_value = max(float(row[value_key]) for row in rows) or 1
    lines = []
    for row in rows:
        value = float(row[value_key])
        width = max(4, value / max_value * 100)
        label = escape(str(row[label_key]))
        shown_value = escape(value_formatter(value))
        lines.append(
            f"""
            <div class="bar-row">
                <div class="bar-label" title="{label}">{label}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: {width:.2f}%"></div>
                </div>
                <div class="bar-value">{shown_value}</div>
            </div>
            """
        )
    return "\n".join(lines)


def kpi_card(label: str, value: str, note: str) -> str:
    return f"""
    <article class="kpi-card">
        <span>{escape(label)}</span>
        <strong>{escape(value)}</strong>
        <small>{escape(note)}</small>
    </article>
    """


def rules_table(rules: pd.DataFrame) -> str:
    if rules.empty:
        return "<p class=\"empty\">Run fpgrowth_analysis.py first to generate association rules.</p>"

    rows = []
    display = rules.head(12).copy()
    for _, row in display.iterrows():
        rows.append(
            f"""
            <tr>
                <td>{escape(str(row["antecedents"]))}</td>
                <td>{escape(str(row["consequents"]))}</td>
                <td>{format_percent(float(row["support"]))}</td>
                <td>{format_percent(float(row["confidence"]))}</td>
                <td>{format_decimal(float(row["lift"]))}x</td>
                <td>{format_decimal(float(row["business_score"]), 4)}</td>
            </tr>
            """
        )

    return f"""
    <table>
        <thead>
            <tr>
                <th>When customer buys</th>
                <th>Recommend</th>
                <th>Support</th>
                <th>Confidence</th>
                <th>Lift</th>
                <th>Business score</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def build_dashboard() -> str:
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    departments = pd.read_csv(DATA_DIR / "departments.csv")
    aisles = pd.read_csv(DATA_DIR / "aisles.csv")

    summary = summarize_order_products()
    product_counts = summary["product_counts"]
    product_reorders = summary["product_reorders"]
    basket_sizes = summary["basket_sizes"]
    total_rows = int(summary["total_rows"])
    total_reorders = int(summary["total_reorders"])

    product_metrics = (
        pd.DataFrame(
            {
                "product_id": product_counts.index.astype(int),
                "order_lines": product_counts.values,
                "reorders": product_reorders.reindex(product_counts.index).fillna(0).values,
            }
        )
        .merge(products, on="product_id", how="left")
        .merge(departments, on="department_id", how="left")
        .merge(aisles, on="aisle_id", how="left")
    )
    product_metrics["reorder_rate"] = (
        product_metrics["reorders"] / product_metrics["order_lines"]
    ).fillna(0)

    department_metrics = (
        product_metrics.groupby("department", as_index=False)
        .agg(order_lines=("order_lines", "sum"), reorders=("reorders", "sum"))
        .sort_values("order_lines", ascending=False)
    )
    department_metrics["reorder_rate"] = (
        department_metrics["reorders"] / department_metrics["order_lines"]
    ).fillna(0)

    if RULES_FILE.exists():
        rules = pd.read_csv(RULES_FILE).sort_values("business_score", ascending=False)
    else:
        rules = pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
                "business_score",
            ]
        )

    orders_by_hour = (
        orders["order_hour_of_day"]
        .value_counts()
        .sort_index()
        .rename_axis("hour")
        .reset_index(name="orders")
    )
    orders_by_hour["hour"] = orders_by_hour["hour"].apply(lambda hour: f"{int(hour):02d}:00")

    orders_by_day = (
        orders["order_dow"]
        .value_counts()
        .sort_index()
        .rename_axis("day")
        .reset_index(name="orders")
    )
    orders_by_day["day"] = orders_by_day["day"].map(DOW_LABELS)

    top_products = (
        product_metrics.sort_values("order_lines", ascending=False)
        .head(10)[["product_name", "order_lines"]]
        .rename(columns={"product_name": "label", "order_lines": "value"})
        .to_dict("records")
    )
    top_departments = (
        department_metrics.head(10)[["department", "order_lines"]]
        .rename(columns={"department": "label", "order_lines": "value"})
        .to_dict("records")
    )
    department_reorders = (
        department_metrics.sort_values("reorder_rate", ascending=False)
        .head(10)[["department", "reorder_rate"]]
        .rename(columns={"department": "label", "reorder_rate": "value"})
        .to_dict("records")
    )
    top_rules = (
        rules.head(10)
        .assign(rule=lambda df: df["antecedents"] + " -> " + df["consequents"])
        [["rule", "business_score"]]
        .rename(columns={"rule": "label", "business_score": "value"})
        .to_dict("records")
        if not rules.empty
        else []
    )

    best_rule = rules.iloc[0] if not rules.empty else None
    best_rule_text = (
        f"{best_rule['antecedents']} -> {best_rule['consequents']}"
        if best_rule is not None
        else "No rules generated yet"
    )

    kpis = [
        kpi_card("Orders", format_number(len(orders)), "All Instacart order records"),
        kpi_card("Customers", format_number(orders["user_id"].nunique()), "Unique shoppers"),
        kpi_card("Order lines", format_number(total_rows), "Product-level transactions"),
        kpi_card("Products", format_number(len(products)), "Catalog size"),
        kpi_card("Departments", format_number(len(departments)), "Product departments"),
        kpi_card("Aisles", format_number(len(aisles)), "Product aisle groups"),
        kpi_card("Avg basket size", format_decimal(float(basket_sizes.mean())), "Items per order"),
        kpi_card("Multi-item baskets", format_number(int((basket_sizes > 1).sum())), "Useful for rules"),
        kpi_card("Reorder rate", format_percent(total_reorders / total_rows), "Repeat purchase share"),
        kpi_card("Association rules", format_number(len(rules)), "FP-Growth rules exported"),
        kpi_card(
            "Avg confidence",
            format_percent(float(rules["confidence"].mean())) if not rules.empty else "0.0%",
            "Rule predictability",
        ),
        kpi_card(
            "Max lift",
            f"{format_decimal(float(rules['lift'].max()))}x" if not rules.empty else "0.00x",
            "Strongest non-random link",
        ),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Instacart Market Basket Dashboard</title>
    <style>
        :root {{
            --bg: #f6f7f2;
            --surface: #ffffff;
            --ink: #1f2933;
            --muted: #64748b;
            --line: #d9e0d2;
            --green: #2f7d57;
            --mint: #dcefe4;
            --gold: #c47f2c;
            --blue: #256b8f;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            background: var(--bg);
            color: var(--ink);
            font-family: Inter, Segoe UI, Arial, sans-serif;
        }}

        header {{
            background: #153b2d;
            color: #fff;
            padding: 34px clamp(18px, 4vw, 54px) 28px;
        }}

        header p {{
            color: #d8eadf;
            margin: 8px 0 0;
            max-width: 860px;
            line-height: 1.55;
        }}

        h1 {{
            font-size: clamp(2rem, 4vw, 3.4rem);
            margin: 0;
            letter-spacing: 0;
        }}

        h2 {{
            font-size: 1.1rem;
            margin: 0 0 16px;
        }}

        main {{
            width: min(1440px, 100%);
            margin: 0 auto;
            padding: 24px clamp(14px, 3vw, 36px) 48px;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(180px, 1fr));
            gap: 14px;
        }}

        .kpi-card,
        .panel {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06);
        }}

        .kpi-card {{
            padding: 18px;
            min-height: 126px;
            display: grid;
            align-content: space-between;
        }}

        .kpi-card span,
        .kpi-card small,
        .section-label {{
            color: var(--muted);
        }}

        .kpi-card strong {{
            font-size: clamp(1.55rem, 3vw, 2.4rem);
            line-height: 1.1;
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 16px;
            margin-top: 16px;
        }}

        .panel {{
            padding: 20px;
        }}

        .span-4 {{
            grid-column: span 4;
        }}

        .span-6 {{
            grid-column: span 6;
        }}

        .span-8 {{
            grid-column: span 8;
        }}

        .span-12 {{
            grid-column: span 12;
        }}

        .insight {{
            background: #f2f8f4;
            border: 1px solid #c9e1d2;
            color: #153b2d;
            padding: 16px;
            border-radius: 8px;
            line-height: 1.55;
        }}

        .bar-row {{
            display: grid;
            grid-template-columns: minmax(120px, 1.2fr) minmax(130px, 3fr) 88px;
            align-items: center;
            gap: 12px;
            min-height: 32px;
            margin: 8px 0;
        }}

        .bar-label {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 0.9rem;
        }}

        .bar-track {{
            height: 12px;
            background: #edf1e8;
            border-radius: 999px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--green), var(--blue));
            border-radius: inherit;
        }}

        .bar-value {{
            color: var(--muted);
            font-variant-numeric: tabular-nums;
            text-align: right;
            font-size: 0.86rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th,
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid var(--line);
            text-align: left;
            vertical-align: top;
        }}

        th {{
            color: #385245;
            background: #eef5ef;
            font-weight: 700;
        }}

        .table-wrap {{
            overflow-x: auto;
        }}

        .empty {{
            color: var(--muted);
            margin: 0;
        }}

        footer {{
            color: var(--muted);
            margin-top: 24px;
            font-size: 0.88rem;
        }}

        @media (max-width: 980px) {{
            .kpi-grid {{
                grid-template-columns: repeat(2, minmax(160px, 1fr));
            }}

            .span-4,
            .span-6,
            .span-8 {{
                grid-column: span 12;
            }}
        }}

        @media (max-width: 620px) {{
            .kpi-grid {{
                grid-template-columns: 1fr;
            }}

            .bar-row {{
                grid-template-columns: 1fr 72px;
            }}

            .bar-track {{
                grid-column: 1 / -1;
                grid-row: 2;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <p class="section-label">Data Mining Project</p>
        <h1>Instacart Market Basket Dashboard</h1>
        <p>Operational summary of catalog coverage, shopping behavior, reorder patterns, and FP-Growth association rules for recommendation and cross-sell decisions.</p>
    </header>

    <main>
        <section class="kpi-grid">
            {''.join(kpis)}
        </section>

        <section class="dashboard-grid">
            <article class="panel span-8">
                <h2>Top Products by Order Lines</h2>
                {bar_table(top_products, "label", "value")}
            </article>

            <article class="panel span-4">
                <h2>Best Rule</h2>
                <div class="insight">
                    <strong>{escape(best_rule_text)}</strong><br>
                    {escape("Highest business score across generated FP-Growth association rules.")}
                </div>
            </article>

            <article class="panel span-6">
                <h2>Top Departments by Demand</h2>
                {bar_table(top_departments, "label", "value")}
            </article>

            <article class="panel span-6">
                <h2>Departments with Highest Reorder Rate</h2>
                {bar_table(department_reorders, "label", "value", format_percent)}
            </article>

            <article class="panel span-6">
                <h2>Orders by Hour</h2>
                {bar_table(orders_by_hour.rename(columns={"hour": "label", "orders": "value"}).to_dict("records"), "label", "value")}
            </article>

            <article class="panel span-6">
                <h2>Orders by Day of Week</h2>
                {bar_table(orders_by_day.rename(columns={"day": "label", "orders": "value"}).to_dict("records"), "label", "value")}
            </article>

            <article class="panel span-12">
                <h2>Top Association Rules by Business Score</h2>
                {bar_table(top_rules, "label", "value", lambda value: format_decimal(value, 4))}
            </article>

            <article class="panel span-12">
                <h2>Association Rule Detail</h2>
                <div class="table-wrap">
                    {rules_table(rules)}
                </div>
            </article>
        </section>

        <footer>
            Generated from Instacart CSV files and fpgrowth_association_rules.csv.
            Re-run <code>python dashboard/generate_dashboard.py</code> after updating the analysis output.
        </footer>
    </main>
</body>
</html>"""


def main() -> None:
    html = build_dashboard()
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
