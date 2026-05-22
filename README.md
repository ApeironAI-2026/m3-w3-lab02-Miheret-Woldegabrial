#1.	Introduction

Sequential data is data where the position (or timestamp) of each element carries information.
RNNs (Recurrent Neural Networks) processes one element at a time and maintains a hidden state that serves as memory:
      x₁                   x₂                     x₃                       x₄
       │                  │                      │                         │
       ▼               ▼                    ▼                       ▼
   ┌──────┐      ┌──────┐      ┌──────┐       ┌──────┐
   │ RNN    │─►│ RNN    │─►│ RNN    │─►│ RNN    │──► output
   │ Cell      │      │ Cell      │      │ Cell      │      │ Cell      │
   └──────┘      └──────┘      └──────┘       └──────┘
   h₀─────►h₁─────►h₂─────►h₃─────►h₄

RNNs are better than MLPs for time-series because they remember past information while MLPs treat every input as independent.

The difference between RNN, LSTM, and GRU 
LSTM adds a cell state (a highway for gradients) and three gates to control information flow: - Forget gate, input gate and Output gate
GRU(Gated Recurrent Unit) is a simplified LSTM with only two gates and no separate cell state: -  Update gate and Reset gate.
Model	Gates	Cell state	Vanishing gradient resistance	Parameters	Best for
RNN	None	No	Weak	Few	Simple/short sequences
LSTM	Forget, Input, Output	Yes	Strong	Most	Long dependencies, complex patterns
GRU	Update, Reset	No (merged into hidden state)	Strong	Medium	Faster training, similar performance to LSTM


┌──────────────────────────────────────────┐
│                LSTM Cell                                                            │
│                                                                                             │
│   ┌─────────┐  ┌─────────┐  ┌────────┐               │
│   │ Forget        │  │  Input         │  │ Output     │             │
│   │  Gate          │  │  Gate          │  │  Gate        │             │
│   │ σ(Wf)         │  │ σ(Wi)          │  │ σ(Wo)       │            │
│   └────┬────┘  └────┬────┘  └───┬────┘            │
│               │                         │                       │                       │
│   C(t-1) ──×── + ──×── C(t)                    │                       │
│        forget          add new               ┌───┘                       │
│        old info  info                             │                                │
│                                                            tanh(C(t)) × ──► h(t)
└───────────────────────────────────┘
2.	Dataset
The Stock Dataset
About Stock Price Data
Stock market data is inherently sequential — each observation is tied to a specific date and depends on preceding observations. We use daily OHLCV data (Open, High, Low, Close, Volume).

Column	Type	Description
Open	float	Price at market open
High	float	Highest price during the day
Low	float	Lowest price during the day
Close	float	Price at market close ← our prediction target
Volume	int	Number of shares traded
We will download Apple Inc. (AAPL) stock data from 2018-01-01 to 2024-01-01 using the yfinance library. A synthetic fallback is provided in case the download fails (e.g., no internet connection).
 
asgn_fig_price_history.png
AAPL shows a strong long term upward trend, with the price rising from roughly $40 → $180+ over the period.
The chart shows several periods of elevated volatility, visible through price swings, moving average divergence, and return distribution width.

3.	Methodology
Data Preprocessing
Why Normalize?
Neural networks train faster and more stably when inputs are on a similar scale. We use MinMaxScaler to map prices to [0, 1].

Critical: the scaler must be fit only on training data — fitting on the full dataset leaks future information into the training process.

Sliding Window Approach
To predict the next day's price, we feed the model a window of the past N days:
Sliding Window (seq_length=60):

Day: 1  2  3  4  ...  60  → predict Day 61
         2  3  4  5  ...  61  → predict Day 62
         3  4  5  6  ...  62  → predict Day 63
         ...
Each window becomes one training sample: input = [price_t-59, ..., price_t], target = price_t+1.
Chronological Split (NOT Random!)
For time-series data, we must never use random train/test splits — that would leak future data into training. Instead, we use a chronological split:

|◄────── Train (80%) ──────►|◄── Test (20%) ──►|
2018                    ~2022              2024
This simulates the real scenario: train on historical data, evaluate on future (unseen) data.
Model Architecture
Architecture Diagram
Our Stock Predictor model supports three recurrent backends — rnn, lstm, or gru — selected by a single string argument:

INPUT: (batch, seq_length, 1) — 60 days of normalized prices
 ┌─── RNN / LSTM / GRU ──────────────────────────────────────┐
 │  x₁ → [h₁] → x₂ → [h₂] → ... → x₆₀ → [h₆₀]                                             │
 │  hidden_size=64, num_layers=2, dropout=0.2                                                 │
 └────────────────────────────────────────────────────────────┘
                         │
              h₆₀ (last hidden state)
                         │
              ┌───   FC Layer    ───┐
              │  Linear(64, 1)           │
              └────────────────┘
                         │
OUTPUT: (batch, 1) — predicted next-day price (scaled)
All three models share the same outer structure — only the recurrent cell differs. This makes comparison fair.
Loss Function & Optimizer
Component	Choice					Explanation
Loss		MSELoss		Mean Squared Error — standard for regression tasks. 
Penalizes large errors more.
Optimizer	Adam			Adaptive learning rate — combines momentum and 
RMSprop. Works well out of the box.
Scheduler	ReduceLROnPlateau	Reduces LR by a factor when validation loss stops 
improving. Prevents overshooting.

4.	Results
 
asgn_fig_predictions_combined.png
 
asgn_fig_predictions_zoom.png
==========================================================================
Metric                          RNN           LSTM            GRU           Best
===========================================================================
MSE                         13.0680        14.7295        16.4630            RNN ★
RMSE                         3.6150          3.8379          4.0575            RNN ★
MAE                           2.8811          3.0449          3.1788            RNN ★
MAPE                         1.7814          1.8997          1.9978            RNN ★
R2                               0.9656          0.9612          0.9567            RNN ★
===========================================================================

Training Time (s)              6.1                 6.5                1.5            GRU ★
Parameters                 12,673          50,497         37,889

 
asgn_fig_metric_comparison.png
╔═════════════════════════════════════════╗
║  BEST MODEL:   RNN                                                                                           ║
║  RMSE:  3.6150                                                                                                    ║
║  MAE:   2.8811                                                                                                     ║
║  MAPE:  1.7814%                                                                                                 ║
║  R²:    0.9656                                                                                                         ║
╚═════════════════════════════════════════╝
5.	Analysis
 
asgn_fig_error_distributions.png
 
asgn_fig_scatter_actual_vs_predicted.png
Best model: RNN (RMSE = 3.6150)

TOP 10 BEST PREDICTIONS (smallest error):
-------------------------------------------------------
  2023-05-25 | Actual: $170.60 | Predicted: $170.60 | Error: $0.00
  2023-07-11 | Actual: $185.48 | Predicted: $185.50 | Error: $0.02
  2023-10-13 | Actual: $176.62 | Predicted: $176.63 | Error: $0.02
  2023-09-14 | Actual: $173.54 | Predicted: $173.51 | Error: $0.03
  2023-03-07 | Actual: $149.30 | Predicted: $149.25 | Error: $0.05
  2022-10-19 | Actual: $141.22 | Predicted: $141.30 | Error: $0.08
  2023-03-08 | Actual: $150.55 | Predicted: $150.63 | Error: $0.08
  2023-02-13 | Actual: $151.51 | Predicted: $151.41 | Error: $0.10
  2023-03-14 | Actual: $150.27 | Predicted: $150.17 | Error: $0.10
  2023-06-06 | Actual: $176.73 | Predicted: $176.62 | Error: $0.11

TOP 10 WORST PREDICTIONS (largest error):
-------------------------------------------------------
  2022-11-03 | Actual: $136.34 | Predicted: $149.40 | Error: $13.06
  2022-11-04 | Actual: $136.07 | Predicted: $145.86 | Error: $9.79
  2022-12-16 | Actual: $132.27 | Predicted: $141.99 | Error: $9.72
  2022-12-15 | Actual: $134.22 | Predicted: $143.66 | Error: $9.44
  2022-12-19 | Actual: $130.16 | Predicted: $139.42 | Error: $9.26
  2022-11-29 | Actual: $138.81 | Predicted: $147.55 | Error: $8.74
  2022-11-02 | Actual: $142.37 | Predicted: $150.99 | Error: $8.62
  2023-08-04 | Actual: $179.47 | Predicted: $187.93 | Error: $8.46
  2023-08-07 | Actual: $176.38 | Predicted: $184.67 | Error: $8.29
  2022-12-28 | Actual: $123.94 | Predicted: $132.10 | Error: $8.16
 
asgn_fig_training_curves.png
GRUs usually converge the fastest, LSTMs are close behind, and RNNs converge the slowest because they struggle with vanishing gradients.
6.	Experiments
Experiment A: Sequence Length
========================================
  seq_length=  20 → RMSE = 4.8806
  seq_length=  40 → RMSE = 4.7472
  seq_length=  60 → RMSE = 5.0241
  seq_length=  90 → RMSE = 4.7968
  seq_length= 120 → RMSE = 4.7944
sequence length has a mild effect on RMSE, with no clear monotonic trend. Performance improves slightly when going from 20 → 40, then fluctuates without a strong upward or downward pattern
 
asgn_fig_exp_seq_length.png
Experiment B: Hidden Size
========================================
  hidden_size=  16 → RMSE = 6.0608
  hidden_size=  32 → RMSE = 5.9629
  hidden_size=  64 → RMSE = 5.1638
  hidden_size= 128 → RMSE = 4.6637
  hidden_size= 256 → RMSE = 3.7736 Larger hidden sizes produce better performance
 
asgn_fig_exp_hidden_size.png
Experiment D: Number of Layers
========================================
  num_layers=1 → RMSE = 4.6052
  num_layers=2 → RMSE = 5.4482
  num_layers=3 → RMSE = 5.4237
 
asgn_fig_exp_num_layers.png
1 layer → RMSE = 4.6052 (best)
2 layers → RMSE = 5.4482 (worse)
3 layers → RMSE = 5.4237 (still worse than 1 layer)

The best setting for each
-	Number of layers: 1 Reason: Lowest RMSE (4.6052); deeper models increased error.
-	Hidden size: 256 Reason: Clear monotonic improvement, best RMSE (3.7736).
-	Sequence length: 40 Reason: Best RMSE (4.7472); longer sequences didn’t consistently help and sometimes hurt.

7.	Conclusion
Across all three models (RNN, LSTM, GRU), the RNN achieved the best overall performance:
•	RMSE: 3.6150
•	MAE: 2.8811
•	MAPE: 1.7814%
•	R²: 0.9656
As stated above:
“Best model: RNN (RMSE = 3.6150)”
