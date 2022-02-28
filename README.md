Example of LSTM application. Two .py files are included:
- data.py with functions to adapt data to the desired learning environment
  - **BuoysFrame** and **ERA5frame** that create DataFrames from txt with multiindex columns: buoyID and time, having specific txt details
  - **target** that creates the target variable(s), which is preciditon bias (model-real) of one or more variables
  - **scaled_dataset** and **split_years** that transform data to desired scale and split train/test based on year cycle, respectively
  - **series_to_supervised** reframes data for LSTMs
- example.py with an example of LSTM application to that data
