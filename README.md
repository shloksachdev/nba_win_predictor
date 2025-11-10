# NBA Win Predictor

A web application that predicts NBA game outcomes using machine learning. Built with NiceGUI, it provides an intuitive interface to view scheduled games, predict winners based on team statistics, and compare key performance metrics.

## Features

- **Game Schedule Display**: View NBA games for any selected date.
- **Win Probability Predictions**: Uses a trained RandomForest model to predict game winners and display win probabilities.
- **Team Statistics Comparison**: Compare top 15 most important stats between teams with color-coded differences.
- **Interactive UI**: Expandable cards showing detailed stats with tooltips for advanced metrics.
- **Team Logos**: Visual representation with official NBA team badges.
- **Date Picker**: Select any date to view predictions for past or future games.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shloksachdev/nba-win-predictor.git
   cd nba-win-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the data files in the `data/csv/` directory and trained model in `model/`.

## Usage

Run the application:
```bash
python main.py
```

Open your browser to `http://localhost:8080` (or the port specified by NiceGUI).

- Select a date using the date picker on the left.
- Click "Predict" to load games and predictions for that date.
- Expand game cards to view detailed statistics comparison.

## Data Preparation

The project uses several Jupyter notebooks for data preparation:

- `data/fetcher.ipynb`: Fetches raw game data and statistics from NBA APIs or sources.
- `data/builder.ipynb`: Processes and builds the dataset from fetched data.
- `data/averager.ipynb`: Calculates team averages and prepares features for the model.

Run these notebooks in order to prepare the CSV files in `data/csv/`:
- `averages.csv`: Team statistics averages
- `schedule.csv`: Game schedules
- `gamelogs.csv`: Individual game logs
- `dataset.csv`: Processed dataset for training
- `results.csv`: Game results

## Model Training

The model is trained using the processed data. The trained RandomForest model is saved as `model/trained_model.pkl`.

To retrain the model:
1. Ensure `data/csv/dataset.csv` and `data/csv/results.csv` are up to date.
2. Run the training script (if available) or modify the training code in the notebooks.

## Dependencies

- Python 3.8+
- nicegui
- pandas
- scikit-learn
- joblib
- numpy

See `requirements.txt` for exact versions.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
