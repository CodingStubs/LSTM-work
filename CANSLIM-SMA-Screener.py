from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
import datetime
import time
import multiprocessing

yf.pdr_override()

# Start the timer
start_time = time.time()


# Function to process a chunk of tickers
def process_chunk(tickers, start_date, end_date, conditions):
    results = []
    for ticker in tickers:
        try:
            if (conditions == 1):
                stock = yf.Ticker(ticker)
                # Page 156, Set a Minimum Level for Current Earnings Increases, paragraph 1
                # In 600 best stocks 1952-2001, 3 of 4 had at least 70% earnings growth YOY in quarter
                # before they made their big move. Those that did not showed earnings growth the next quarter
                # with average increases of 90%
                eps = stock.income_stmt.loc["Basic EPS"]
                YOY_eps_growth = ((eps.iloc[0] / eps.iloc[1]) - 1) * 100
                condition_C0 = YOY_eps_growth > 20
                #Stop screening if C0 not met
                if (condition_C0):
                    quarterly_eps = stock.quarterly_income_stmt.loc["Basic EPS"]
                    quarterly_eps["Q1Q2 Change"] = ((quarterly_eps.iloc[0] / quarterly_eps[1]) - 1) * 100
                    quarterly_eps["Q2Q3 Change"] = ((quarterly_eps.iloc[1] / quarterly_eps[2]) - 1) * 100
        
                    # Page 156, Set a Minimum Level for Current Earnings Increases, paragraph 2
                    # (modified numbers from 40% - 500% down to 20%)
                    condition_C1 = (quarterly_eps["Q1Q2 Change"] >= 20 and quarterly_eps["Q2Q3 Change"] >= 20)
                    
                    revenue = stock.quarterly_income_stmt.loc["Total Revenue"]
                    revenue["Q1Q2 Change"] = ((revenue.iloc[0] / revenue[1]) - 1) * 100
                    
                    # Page 158, Insist on Sales Growth as Well as Earnings Growth, paragraph 1
                    condition_C2 = revenue["Q1Q2 Change"] > 25
        
                    print(f"Ticker: {ticker} Condition: {condition_C1}\n")
                    time.sleep(1)
                    
                    if (condition_C1 and condition_C2):
                        results.append((ticker, YOY_eps_growth, condition_C1, condition_C2))
                    
            elif (conditions == 2):
                df = pdr.get_data_yahoo(ticker, start_date, end_date)
                df.to_csv(f'{ticker}.csv')
                    
                # Get data from CSV                
                df = pd.read_csv(f'{ticker}.csv')

                # Calculate different length SMA
                sma = [50, 150, 200]
                for x in sma:
                    df["SMA_" + str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)

                # Create variables to reference in conditions
                currentClose = df["Adj Close"].iloc[-1]
                moving_average_50 = df["SMA_50"].iloc[-1]
                moving_average_150 = df["SMA_150"].iloc[-1]
                moving_average_200 = df["SMA_200"].iloc[-1]
                low_of_52week = round(min(df["Low"].iloc[-260:]), 2)
                high_of_52week = round(max(df["High"].iloc[-260:]), 2)

                try:
                    moving_average_200_20 = df["SMA_200"].iloc[-20]
                except Exception:
                    moving_average_200_20 = 0

                # Condition 1: Current Price > 150 SMA and > 200 SMA
                condition_1 = currentClose > moving_average_150 and currentClose > moving_average_200

                # Condition 2: 150 SMA and > 200 SMA
                condition_2 = moving_average_150 > moving_average_200

                # Condition 3: 200 SMA trending up for at least 1 month
                condition_3 = moving_average_200 > moving_average_200_20

                # Condition 4: 50 SMA > 150 SMA and 50 SMA > 200 SMA
                condition_4 = moving_average_50 > moving_average_150 and moving_average_50 > moving_average_200

                # Condition 5: Current Price > 50 SMA
                condition_5 = currentClose > moving_average_50

                # Condition 6: Current Price is at least 30% above 52-week low
                condition_6 = currentClose >= (1.3 * low_of_52week)

                # Condition 7: Current Price is within 25% of 52-week high
                condition_7 = currentClose >= (.75 * high_of_52week)


                # If all conditions above are true, add stock to results
                if condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7:
                    results.append((ticker, moving_average_50, moving_average_150, moving_average_200))

        except Exception as e:
            print(e)
            print(f"Could not gather data on {ticker}")

    return results

def get_index():
    print("Select an Index of Stocks to Screen:")
    print("1. S&P500")
    print("2. Dow Jones Industrial Average")
    print("3. NASDAQ")
    
    index = int(input("Enter your choice (1/2/3): "))
    if (index == 1):
        return si.tickers_sp500()
    elif (index == 2):
        return si.tickers_dow()
    elif (index == 3):
        return si.tickers_nasdaq()
    else:
        return get_index()
    
def get_conditions():
    print("\nSelect a preset condition set:")
    print("1. CANSLIM")
    print("2. MINERVINI")
    conditions = int(input("Enter your choice (1/2): "))
    if (conditions == 1 or conditions == 2):
        return conditions
    else:
        return get_conditions()
    
if __name__ == '__main__':
    # Number of processes to run in parallel
    num_processes = multiprocessing.cpu_count()
        
    # Variables
    tickers = get_index()
    tickers = [item.replace(".", "-") for item in tickers]  # Yahoo Finance uses dashes instead of dots
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    stock_list = []

    # Split the tickers into smaller chunks for multiprocessing
    chunk_size = len(tickers) // num_processes
    ticker_chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]

    # Create a Pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Process the ticker chunks in parallel and collect the results
    conditions = get_conditions()
    results = pool.starmap(process_chunk, [(chunk, start_date, end_date, conditions) for chunk in ticker_chunks])

    # Merge the results from each process
    for result in results:
        stock_list.extend(result)
        
    if (conditions == 1):
        stock_list = pd.DataFrame(stock_list, columns=['Ticker', 'YOY EPS % Growth', 'Quarterly Earnings Check', 'Revenue Growth Check'])
    if (conditions == 2):
        stock_list = pd.DataFrame(stock_list, columns=['Ticker', '50 day MA', '150 day MA', '200 day MA'])
    print("\n")
    print(stock_list)

    # Print the total execution time
    print("\n\nScreening finished. Stocks listed under variable 'stock_list'\n")
    print(f"Execution Time: {(time.time() - start_time):.2f} seconds")
