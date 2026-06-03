# Titanic Advanced BI & Predictive Machine Learning Suite

This directory contains a Streamlit-powered Business Intelligence (BI) and Machine Learning (ML) dashboard application ([dashboard.py](dashboard.py)) designed to explore historical passenger demographics, survival statistics, and run real-time predictive simulations on the Titanic passenger dataset ([titanic_cleaned.csv](titanic_cleaned.csv)).

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
  - [1. Data Engine & Feature Engineering](#1-data-engine--feature-engineering)
  - [2. Interactive Controls & Dynamic Metrics](#2-interactive-controls--dynamic-metrics)
  - [3. Advanced Interactive Visualizations (Plotly)](#3-advanced-interactive-visualizations-plotly)
  - [4. AI Survival Simulator (Random Forest Classifier)](#4-ai-survival-simulator-random-forest-classifier)
- [Installation and Dependencies](#installation-and-dependencies)
- [How to Run the App](#how-to-run-the-app)

---

## Overview

The Titanic Advanced BI & Predictive Machine Learning Suite ([dashboard.py](dashboard.py)) is a high-fidelity data space that bridges the gap between historical data analysis and predictive modeling. Built using **Streamlit**, **Pandas**, **Plotly Express**, and **Scikit-learn**, it allows users to filter population segments, visualize demographic pathways, and run live machine learning inferences on custom profiles.

---

## Key Features

### 1. Data Engine & Feature Engineering
Upon loading the dataset ([titanic_cleaned.csv](titanic_cleaned.csv)), the app performs on-the-fly feature engineering:
- **Survival Status**: Maps binary `is_survived` to descriptive labels (`Survived` / `Died`).
- **Ticket Class**: Maps `passenger_class` numbers to text descriptions (`1st Class`, `2nd Class`, `3rd Class`).
- **Embarkation Port**: Reconstructs complete port names from single-character codes (`Cherbourg`, `Southampton`, `Queenstown`).
- **Family Size**: Calculates total family size as `siblings_spouses_count + parents_children_count`.
- **Age Brackets**: Segments age values into groups (`Child`, `Teenager`, `Young Adult`, `Adult`, `Senior`).
- **Deck Extraction**: Extracts deck level codes from the cabin code values.
- **Title Extraction**: Extracts social and noble titles (e.g., `Mr.`, `Mrs.`, `Dr.`, `Sir.`) from passenger names.

### 2. Interactive Controls & Dynamic Metrics
The sidebar offers real-time filters:
- **Search Passenger Name**: Text search to find specific passengers.
- **Gender Selection**: Multiselect checkboxes for male and female passengers.
- **Class Tier Selection**: Multiselect for 1st, 2nd, and 3rd Class.
- **Ship Deck Filter**: Multiselect for cabin decks (A through G, T, Unknown).

A metrics ribbon at the top dynamically updates to display:
- **Filtered Population**: Total headcount matching selected filters.
- **Survival Probability**: The percentage rate of survival in the filtered cohort.
- **Average Fare Cost**: Average price paid for tickets in the filtered cohort.
- **Dominant Social Demographic**: The most common passenger title (mode value).

### 3. Advanced Interactive Visualizations (Plotly)
- **Demographic Survival Ribbons Flow**: A parallel categories chart tracing structural flows of survival lines across ticket tiers and gender classifications.
- **Survival Probability Hotspots**: A density heatmap highlighting concentration counts of survivors across class tiers and age brackets.
- **Financial Fare Density Splitting**: An overlay histogram showcasing price distribution overlays split by survival outcome.
- **Survival Proportions by Ship Deck Level**: A proportional bar chart showing survival percentage ratios relative to deck levels.

### 4. AI Survival Simulator (Random Forest Classifier)
- **Model Details**: Runs a trained Scikit-learn `RandomForestClassifier` in the backend, utilizing features like class, gender, age, fare, family count, and boarding port.
- **Interactive Inputs**: Interactive fields (sliders, drop-downs, buttons) allow users to construct custom passenger profiles.
- **Live Inference**: Renders predicted survival outcomes (SURVIVED or DIED) along with confidence score percentages in real-time.
- **Model Drivers**: Displays a horizontal bar chart of the model's global feature importances, showing which attributes the AI prioritized most during training.

---

## Installation and Dependencies

Ensure you have the required packages installed before running the application:

```bash
pip install streamlit pandas plotly scikit-learn
```

---

## How to Run the App

1. Ensure the clean dataset file ([titanic_cleaned.csv](titanic_cleaned.csv)) is in the same directory as the script.
2. Open your terminal and run the following command:
   ```bash
   streamlit run dashboard.py
   ```
3. A local web browser page will open automatically at `http://localhost:8501`.
