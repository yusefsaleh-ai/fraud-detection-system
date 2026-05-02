# AI-Based Financial Fraud Detection System

This project is a full-stack fraud detection system that integrates machine learning and blockchain technology to detect and securely log financial transactions.

The system uses a synthetic UK-based financial transaction dataset for training and evaluation.

## Features
- Real-time fraud prediction using a Random Forest model
- Flask API for handling prediction requests
- Web-based frontend for user interaction
- Blockchain-inspired ledger for secure and immutable transaction storage

## Project Structure
- api/ – Flask backend
- model/ – Machine learning model and training scripts
- frontend/ – User interface
- blockchain/ – Blockchain implementation
- data/ – Dataset and generation scripts

## Technologies Used
- Python
- Flask
- Scikit-learn
- HTML & JavaScript

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Flask app:
   python api/app.py

3. Open in browser:
   http://127.0.0.1:5000/

## Demo

The system allows users to input transaction details via a web interface and receive real-time fraud predictions. Each transaction is securely recorded in a blockchain-inspired ledger.

## Author
Yusef Saleh
