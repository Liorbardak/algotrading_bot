import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns

# Sample data structure - replace with your actual data
sample_data = {
    '2024-01-01': ['AAPL', 'GOOGL', 'MSFT'],
    '2024-01-05': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
    '2024-01-10': ['AAPL', 'GOOGL', 'TSLA', 'AMZN'],
    '2024-01-15': ['GOOGL', 'TSLA', 'AMZN', 'META'],
    '2024-01-20': ['TSLA', 'AMZN', 'META', 'NVDA'],
    '2024-01-25': ['AMZN', 'META', 'NVDA', 'NFLX'],
    '2024-01-30': ['META', 'NVDA', 'NFLX', 'AAPL']
}


def create_holdings_heatmap(holdings_dict):
    """Create a heatmap showing stock holdings over time"""
    # Convert to DataFrame
    dates = list(holdings_dict.keys())
    all_stocks = set()
    for stocks in holdings_dict.values():
        all_stocks.update(stocks)

    # Create binary matrix
    matrix = []
    for date in dates:
        row = [1 if stock in holdings_dict[date] else 0 for stock in sorted(all_stocks)]
        matrix.append(row)

    df = pd.DataFrame(matrix, index=dates, columns=sorted(all_stocks))

    plt.figure(figsize=(12, 8))
    sns.heatmap(df, cmap='RdYlGn', cbar_kws={'label': 'Holding Status'},
                yticklabels=True, xticklabels=True, annot=False)
    plt.title('Stock Holdings Over Time (Heatmap)')
    plt.xlabel('Stocks')
    plt.ylabel('Dates')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def create_holdings_timeline(holdings_dict):
    """Create a timeline visualization with stock names"""
    fig, ax = plt.subplots(figsize=(15, 8))

    dates = list(holdings_dict.keys())
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Plot each stock as a horizontal line segment
    all_stocks = set()
    for stocks in holdings_dict.values():
        all_stocks.update(stocks)

    stock_colors = plt.cm.Set3(np.linspace(0, 1, len(all_stocks)))
    color_map = dict(zip(sorted(all_stocks), stock_colors))

    y_pos = 0
    for i, (date, stocks) in enumerate(holdings_dict.items()):
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        for j, stock in enumerate(stocks):
            ax.barh(y_pos, 1, left=i, height=0.8,
                    color=color_map[stock], alpha=0.7,
                    label=stock if stock not in ax.get_legend_handles_labels()[1] else "")
            ax.text(i + 0.5, y_pos, stock, ha='center', va='center',
                    fontsize=8, fontweight='bold')
            y_pos += 1
        y_pos += 0.5  # Add space between dates

    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45)
    ax.set_ylabel('Holdings')
    ax.set_xlabel('Dates')
    ax.set_title('Stock Holdings Timeline')
    ax.grid(True, alpha=0.3)

    # Remove duplicate labels and create legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def create_stock_gantt_chart(holdings_dict):
    """Create a Gantt chart showing when each stock was held"""
    fig, ax = plt.subplots(figsize=(15, 8))

    dates = list(holdings_dict.keys())
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Get all unique stocks
    all_stocks = set()
    for stocks in holdings_dict.values():
        all_stocks.update(stocks)
    all_stocks = sorted(all_stocks)

    # Create color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_stocks)))

    # For each stock, find continuous periods where it was held
    for i, stock in enumerate(all_stocks):
        held_dates = []
        for j, date in enumerate(dates):
            if stock in holdings_dict[date]:
                held_dates.append(j)

        # Group consecutive dates
        if held_dates:
            groups = []
            current_group = [held_dates[0]]

            for k in range(1, len(held_dates)):
                if held_dates[k] == held_dates[k - 1] + 1:
                    current_group.append(held_dates[k])
                else:
                    groups.append(current_group)
                    current_group = [held_dates[k]]
            groups.append(current_group)

            # Plot each group as a horizontal bar
            for group in groups:
                start_idx = group[0]
                duration = len(group)
                ax.barh(i, duration, left=start_idx, height=0.6,
                        color=colors[i], alpha=0.8, label=stock)

    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45)
    ax.set_yticks(range(len(all_stocks)))
    ax.set_yticklabels(all_stocks)
    ax.set_xlabel('Dates')
    ax.set_ylabel('Stocks')
    ax.set_title('Stock Holdings Gantt Chart')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()


def create_simple_text_plot(holdings_dict):
    """Simple text-based visualization"""
    fig, ax = plt.subplots(figsize=(15, 6))

    dates = list(holdings_dict.keys())

    for i, (date, stocks) in enumerate(holdings_dict.items()):
        # Plot vertical line for each date
        ax.axvline(x=i, color='lightgray', linestyle='--', alpha=0.5)

        # Add stock names as text
        stock_text = ', '.join(stocks)
        ax.text(i, 0.5, stock_text, rotation=90, ha='center', va='bottom',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Holdings')
    ax.set_xlabel('Dates')
    ax.set_title('Stock Holdings Over Time')
    ax.grid(True, alpha=0.3)

    # Remove y-axis ticks as they're not meaningful
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Replace 'sample_data' with your actual holdings dictionary
    # Format: {'YYYY-MM-DD': ['STOCK1', 'STOCK2', ...]}

    print("Creating different visualizations of your stock holdings...")

    # 1. Heatmap view
    print("\n1. Heatmap visualization:")
    create_holdings_heatmap(sample_data)

    # 2. Timeline view
    print("\n2. Timeline visualization:")
    create_holdings_timeline(sample_data)

    # 3. Gantt chart view
    print("\n3. Gantt chart visualization:")
    create_stock_gantt_chart(sample_data)

    # 4. Simple text plot
    print("\n4. Simple text visualization:")
    create_simple_text_plot(sample_data)

# Instructions for using with your data:
"""
To use this with your actual data, replace the sample_data dictionary with your data in this format:

your_holdings = {
    '2024-01-01': ['AAPL', 'GOOGL', 'MSFT'],
    '2024-01-02': ['AAPL', 'GOOGL', 'TSLA'],
    # ... more dates and holdings
}

Then call any of the visualization functions:
create_holdings_heatmap(your_holdings)
create_holdings_timeline(your_holdings)
create_stock_gantt_chart(your_holdings)
create_simple_text_plot(your_holdings)
"""