"""Financial analysis tools — ratio calculations and investment assessment.

Provides `financial_analyzer` tool for the agent to compute key financial
ratios from structured data extracted via RAG or user input.

Ratios computed:
- Profitability: ROE, ROA, Net Profit Margin, Gross Profit Margin
- Leverage: DER, DAR
- Liquidity: Current Ratio, Quick Ratio
- Efficiency: Asset Turnover
- Growth: YoY Revenue/Profit growth
"""
from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _parse_number(value) -> float | None:
    """Parse a number from various formats (string with units, plain number)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove common formatting
        cleaned = value.strip().upper()
        cleaned = cleaned.replace(",", "").replace(" ", "")

        multiplier = 1.0
        for suffix, mult in [
            ("T", 1e12), ("TRILIUN", 1e12), ("TRILLION", 1e12),
            ("B", 1e9), ("MILIAR", 1e9), ("BILLION", 1e9),
            ("M", 1e6), ("JUTA", 1e6), ("MILLION", 1e6),
            ("K", 1e3), ("RIBU", 1e3), ("THOUSAND", 1e3),
        ]:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                multiplier = mult
                break

        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None
    return None


def _health_label(value: float, good_threshold: float, bad_threshold: float, higher_is_better: bool = True) -> str:
    """Return a health label: ✅ Sehat, ⚠️ Perlu Perhatian, 🔴 Bahaya."""
    if higher_is_better:
        if value >= good_threshold:
            return "✅ Sehat"
        elif value >= bad_threshold:
            return "⚠️ Perlu Perhatian"
        else:
            return "🔴 Bahaya"
    else:
        if value <= good_threshold:
            return "✅ Sehat"
        elif value <= bad_threshold:
            return "⚠️ Perlu Perhatian"
        else:
            return "🔴 Bahaya"


@tool
def financial_analyzer(data: str) -> str:
    """Analyze financial data and compute key financial ratios for investment assessment.

    Input `data` must be a JSON string with financial figures. Accepted keys:
    - revenue (or pendapatan, sales)
    - cogs (or hpp, cost_of_goods_sold)
    - gross_profit (or laba_kotor)
    - net_income (or laba_bersih, net_profit)
    - total_assets (or total_aset)
    - total_equity (or total_ekuitas, equity)
    - total_debt (or total_utang, total_liabilities)
    - current_assets (or aset_lancar)
    - current_liabilities (or liabilitas_lancar, utang_lancar)
    - inventory (or persediaan)
    - prev_revenue (previous year revenue for growth calc)
    - prev_net_income (previous year net income for growth calc)
    - stock_price (or harga_saham)
    - shares_outstanding (or jumlah_saham)

    Values can include units like "500M", "1.2B", "300 Juta", "1 Triliun".

    Returns a detailed analysis with all computed ratios, health assessment,
    and investment recommendation in a markdown table format.
    """
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return "Error: Input must be valid JSON. Example: {\"revenue\": \"500M\", \"net_income\": \"100M\", \"total_assets\": \"1B\", \"total_equity\": \"600M\"}"

    # Normalize keys (support Indonesian and English aliases)
    key_map = {
        "pendapatan": "revenue", "sales": "revenue", "penjualan": "revenue",
        "hpp": "cogs", "cost_of_goods_sold": "cogs", "beban_pokok": "cogs",
        "laba_kotor": "gross_profit",
        "laba_bersih": "net_income", "net_profit": "net_income",
        "total_aset": "total_assets",
        "total_ekuitas": "total_equity", "equity": "total_equity",
        "total_utang": "total_debt", "total_liabilities": "total_debt",
        "liabilities": "total_debt",
        "aset_lancar": "current_assets",
        "liabilitas_lancar": "current_liabilities", "utang_lancar": "current_liabilities",
        "persediaan": "inventory",
        "harga_saham": "stock_price",
        "jumlah_saham": "shares_outstanding",
    }

    normalized: dict[str, float | None] = {}
    for k, v in parsed.items():
        key = key_map.get(k.lower(), k.lower())
        normalized[key] = _parse_number(v)

    # Extract values
    revenue = normalized.get("revenue")
    cogs = normalized.get("cogs")
    gross_profit = normalized.get("gross_profit")
    net_income = normalized.get("net_income")
    total_assets = normalized.get("total_assets")
    total_equity = normalized.get("total_equity")
    total_debt = normalized.get("total_debt")
    current_assets = normalized.get("current_assets")
    current_liabilities = normalized.get("current_liabilities")
    inventory = normalized.get("inventory")
    prev_revenue = normalized.get("prev_revenue")
    prev_net_income = normalized.get("prev_net_income")
    stock_price = normalized.get("stock_price")
    shares = normalized.get("shares_outstanding")

    # Derive missing values where possible
    if gross_profit is None and revenue and cogs:
        gross_profit = revenue - cogs
    if total_debt is None and total_assets and total_equity:
        total_debt = total_assets - total_equity
    if total_equity is None and total_assets and total_debt:
        total_equity = total_assets - total_debt

    # Compute ratios
    ratios: list[dict] = []

    def _add(category: str, name: str, value: float | None, fmt: str, good: float, bad: float, higher: bool = True):
        if value is not None:
            label = _health_label(value, good, bad, higher)
            ratios.append({
                "category": category,
                "name": name,
                "value": value,
                "formatted": fmt.format(value),
                "health": label,
            })

    # Profitability
    if net_income and total_equity and total_equity != 0:
        _add("Profitabilitas", "ROE (Return on Equity)", net_income / total_equity * 100, "{:.2f}%", 15, 8)
    if net_income and total_assets and total_assets != 0:
        _add("Profitabilitas", "ROA (Return on Assets)", net_income / total_assets * 100, "{:.2f}%", 5, 2)
    if net_income and revenue and revenue != 0:
        _add("Profitabilitas", "Net Profit Margin", net_income / revenue * 100, "{:.2f}%", 10, 5)
    if gross_profit and revenue and revenue != 0:
        _add("Profitabilitas", "Gross Profit Margin", gross_profit / revenue * 100, "{:.2f}%", 30, 15)

    # Leverage
    if total_debt and total_equity and total_equity != 0:
        _add("Leverage", "DER (Debt to Equity)", total_debt / total_equity, "{:.2f}x", 1.0, 2.0, higher_is_better=False)
    if total_debt and total_assets and total_assets != 0:
        _add("Leverage", "DAR (Debt to Asset)", total_debt / total_assets, "{:.2f}x", 0.5, 0.7, higher_is_better=False)

    # Liquidity
    if current_assets and current_liabilities and current_liabilities != 0:
        _add("Likuiditas", "Current Ratio", current_assets / current_liabilities, "{:.2f}x", 1.5, 1.0)
    if current_assets and inventory and current_liabilities and current_liabilities != 0:
        _add("Likuiditas", "Quick Ratio", (current_assets - inventory) / current_liabilities, "{:.2f}x", 1.0, 0.5)

    # Efficiency
    if revenue and total_assets and total_assets != 0:
        _add("Efisiensi", "Asset Turnover", revenue / total_assets, "{:.2f}x", 1.0, 0.5)

    # Growth
    if revenue and prev_revenue and prev_revenue != 0:
        growth = (revenue - prev_revenue) / prev_revenue * 100
        _add("Pertumbuhan", "Revenue Growth (YoY)", growth, "{:.2f}%", 10, 0)
    if net_income and prev_net_income and prev_net_income != 0:
        growth = (net_income - prev_net_income) / prev_net_income * 100
        _add("Pertumbuhan", "Profit Growth (YoY)", growth, "{:.2f}%", 10, 0)

    # Valuation
    if stock_price and shares and net_income and net_income != 0:
        eps = net_income / shares
        per = stock_price / eps
        _add("Valuasi", "PER (Price to Earnings)", per, "{:.2f}x", 15, 25, higher_is_better=False)
    if stock_price and shares and total_equity and total_equity != 0:
        bvps = total_equity / shares
        pbv = stock_price / bvps
        _add("Valuasi", "PBV (Price to Book)", pbv, "{:.2f}x", 1.5, 3.0, higher_is_better=False)

    if not ratios:
        return "Error: Could not compute any ratios. Please provide at least: revenue, net_income, total_assets, total_equity."

    # Build markdown table
    lines = [
        "## 📊 Analisis Rasio Keuangan\n",
        "| Kategori | Rasio | Nilai | Status |",
        "|:---------|:------|------:|:-------|",
    ]
    for r in ratios:
        lines.append(f"| {r['category']} | {r['name']} | {r['formatted']} | {r['health']} |")

    # Investment assessment
    healthy_count = sum(1 for r in ratios if "Sehat" in r["health"])
    warning_count = sum(1 for r in ratios if "Perhatian" in r["health"])
    danger_count = sum(1 for r in ratios if "Bahaya" in r["health"])
    total_count = len(ratios)

    score = (healthy_count * 3 + warning_count * 1) / (total_count * 3) * 100

    if score >= 70:
        recommendation = "🟢 **LAYAK INVESTASI** — Fundamental perusahaan secara keseluruhan sehat."
    elif score >= 40:
        recommendation = "🟡 **PERLU ANALISIS LEBIH LANJUT** — Ada beberapa area yang perlu diperhatikan."
    else:
        recommendation = "🔴 **RISIKO TINGGI** — Fundamental perusahaan menunjukkan kelemahan signifikan."

    lines.extend([
        "",
        "## 💡 Assessment Investasi\n",
        f"- Skor Fundamental: **{score:.0f}/100**",
        f"- Rasio Sehat: {healthy_count}/{total_count}",
        f"- Perlu Perhatian: {warning_count}/{total_count}",
        f"- Bahaya: {danger_count}/{total_count}",
        f"- Rekomendasi: {recommendation}",
    ])

    # Return as structured JSON for Excel generation, wrapped in markdown
    result_json = json.dumps({
        "ratios": ratios,
        "score": round(score, 1),
        "healthy": healthy_count,
        "warning": warning_count,
        "danger": danger_count,
        "total": total_count,
        "recommendation": recommendation,
        "input_data": {k: v for k, v in normalized.items() if v is not None},
    }, ensure_ascii=False)

    lines.extend([
        "",
        f"<!-- FINANCIAL_DATA:{result_json} -->",
    ])

    return "\n".join(lines)


__all__ = ["financial_analyzer"]
