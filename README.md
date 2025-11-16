# Statistical_Arbitrage

A Python‑based framework for implementing statistical arbitrage strategies.  
This repository provides code and structure to support end‑to‑end workflow: data ingestion, strategy modelling, back‑testing, and results analysis.

## Table of Contents

- [Motivation](#motivation)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Configuration](#configuration)  
- [Contributing](#contributing)  
- [License](#license)  

## Motivation

Statistical arbitrage (StatArb) strategies exploit temporary mispricings between related financial instruments — e.g., pairs trading, cointegration‑based spreads, mean‑reversion signals.  
This project aims to provide a clean, modular code‑base for exploring such strategies with Python, enabling data scientists and quants to:  
- ingest and preprocess market data  
- design and test spread‑based trading logic  
- perform back‑tests with configurable parameters  
- generate insights and visualizations of strategy performance  

## Features

- Modular code for strategy definition and back‑testing  
- Support for common statistical arbitrage methods (e.g., spread construction, mean‑reversion triggers)  
- Logging and results output to compare strategy variants  
- Easily extendable to other assets, trading rules or data sources  

## Project Structure

/
├── src/ # Source code for strategy modules, utilities
├── run_project1.py # Entry point script to execute the strategy
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This file


**Key files/folders:**

- `src/` – Contains the core modules (data ingestion, signal generation, back‑testing engine).  
- `run_project1.py` – A sample execution script demonstrating how to run the strategy end‑to‑end.  
- `requirements.txt` – Lists the Python packages required (e.g., pandas, numpy, statsmodels, matplotlib).  
- `.gitignore` – Standard ignore file for Python projects (virtual environments, data caches, etc.).  

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/sohanrajr-sys/Statistical_Arbitrage.git
    cd Statistical_Arbitrage
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate     # On Windows: venv\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To execute the sample strategy provided:

```bash
python run_project1.py
