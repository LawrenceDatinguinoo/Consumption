# Electricity Consumption Analysis

A Python project for Linear Algebra for AI. It uses appliance-related data
to demonstrate matrix operations, PCA, and ordinary least squares regression.

## Dataset

The project uses 1,000 records from the Indian Household Electricity
Consumption Dataset on Kaggle.

Source: https://www.kaggle.com/datasets/suraj520/indian-household-electricity-bill

Predictors: Fan, Refrigerator, AirConditioner, Television, and Monitor.

**Important:** The KWH target is a calculated proxy that reproduces
MonthlyHours. It is not a verified measurement of electricity consumption.

## Installation

Open a terminal in the folder containing `main.py`, then run:

```bash
py -m pip install -r requirements.txt

## run program
py main.py --input "Dataset - Sheet1.csv"

