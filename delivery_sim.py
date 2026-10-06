import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Create 100 synthetic real delivery times
# --------------------------------------------------

np.random.seed(42)

delivery_times = np.random.normal(
    loc=32,
    scale=7,
    size=100
)

delivery_times = np.round(
    delivery_times,
    1
)

# Add three unusually long deliveries
delivery_times[[67, 70, 74]] = [
    68, 71, 75
]

# Make sure delivery times are positive
delivery_times = np.maximum(
    delivery_times,
    5
)

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

data = {
    "OrderID": np.arange(1, 101),
    "DeliveryTime(min)": delivery_times
}

df = pd.DataFrame(data)

# Save CSV
df.to_csv(
    "delivery_times.csv",
    index=False
)

# Load CSV
df = pd.read_csv(
    "delivery_times.csv"
)

# --------------------------------------------------
# A. Mean and standard deviation
# --------------------------------------------------

mean_time = df["DeliveryTime(min)"].mean()

std_time = df["DeliveryTime(min)"].std()

print(
    f"Real delivery times -- "
    f"Mean: {mean_time:.2f} min "
    f"Std: {std_time:.2f} min"
)

# --------------------------------------------------
# B. Simulate expected delivery times
# --------------------------------------------------

simulated = np.random.normal(
    loc=mean_time,
    scale=std_time,
    size=100
)

print(
    f"Simulated 100 values from "
    f"Normal(mean={mean_time:.2f}, "
    f"std={std_time:.2f})"
)

# --------------------------------------------------
# C. Compare real vs simulated
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["DeliveryTime(min)"],
    bins=12,
    alpha=0.6,
    label="Real Delivery Times",
    edgecolor="black"
)

plt.hist(
    simulated,
    bins=12,
    alpha=0.6,
    label="Simulated Normal",
    edgecolor="black"
)

plt.xlabel(
    "Delivery Time (minutes)"
)

plt.ylabel("Frequency")

plt.title(
    "Real vs Simulated Delivery Times"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "delivery_comparison_histogram.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# D. Missing values
# --------------------------------------------------

missing_values = (
    df["DeliveryTime(min)"]
    .isnull()
    .sum()
)

print(
    "\nMissing values:",
    missing_values
)

# --------------------------------------------------
# Percentiles
# --------------------------------------------------

Q1 = df["DeliveryTime(min)"].quantile(0.25)

Q2 = df["DeliveryTime(min)"].quantile(0.50)

Q3 = df["DeliveryTime(min)"].quantile(0.75)

print(
    f"Percentiles -- "
    f"25th: {Q1:.2f} "
    f"50th: {Q2:.2f} "
    f"75th: {Q3:.2f}"
)

# --------------------------------------------------
# Boxplot
# --------------------------------------------------

plt.figure(figsize=(6, 4))

plt.boxplot(
    df["DeliveryTime(min)"]
)

plt.ylabel(
    "Delivery Time (minutes)"
)

plt.title(
    "Delivery Time Boxplot"
)

plt.tight_layout()

plt.savefig(
    "delivery_boxplot.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# E. IQR outlier detection
# --------------------------------------------------

IQR = Q3 - Q1

lower_bound = (
    Q1 - 1.5 * IQR
)

upper_bound = (
    Q3 + 1.5 * IQR
)

outlier_mask = (
    (df["DeliveryTime(min)"] < lower_bound)
    |
    (df["DeliveryTime(min)"] > upper_bound)
)

outliers = df[outlier_mask]

print(
    f"\nIQR outlier bounds: "
    f"{lower_bound:.2f} - "
    f"{upper_bound:.2f}"
)

print(
    f"Outliers detected: "
    f"{len(outliers)} deliveries "
    f"({outliers['DeliveryTime(min)'].tolist()})"
)

percentage = (
    len(outliers) /
    len(df) *
    100
)

print(
    f"Percentage flagged as outliers: "
    f"{percentage:.1f}%"
)

print(
    "\nSaved "
    "delivery_comparison_histogram.png, "
    "delivery_boxplot.png"
)