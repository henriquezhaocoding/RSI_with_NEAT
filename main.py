import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import neat
import os

###
# STEP 1 - IMPORT RAW DATA
###

def get_data(tickers, start, end, interval="1m"):
    # Create an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Fetch and combine historical stock data for each ticker
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start, end=end, interval=interval)
        stock_data['Ticker'] = ticker  # Add a 'Ticker' column to identify the source
        combined_df = pd.concat([combined_df, stock_data])

    return combined_df

###
# STEP 2 - CREATE THE CSV WITH RSI INFORMATION
###

# Choose sample stock
def create_data(combined_df, sample):
    sample_df = combined_df[combined_df["Ticker"] == sample]

    #Calculate moving averages
    sample_df = sample_df.copy()
    sample_df.loc[:, 'MA 15'] = sample_df['Adj Close'].shift(15).rolling(window=15, min_periods=1).mean()

    #Set first 14 rows to NaN
    sample_df.loc[:15, ["RSI", "RS", "Avg Gain", "Avg Loss"]] = np.nan

    #Calculate change
    sample_df.loc[:, "Change"] = sample_df["Adj Close"] - sample_df["Adj Close"].shift(1)
    sample_df.loc[:, "Gain"] = sample_df["Change"].apply(lambda x: x if x >= 0 else 0)
    sample_df.loc[:, "Avg Gain"] = sample_df["Gain"].shift(1).rolling(window=14, min_periods=1).mean()
    sample_df.loc[:, "Loss"] = sample_df["Change"].apply(lambda x: abs(x) if x <= 0 else 0)
    sample_df.loc[:, "Avg Loss"] = sample_df["Loss"].shift(1).rolling(window=14, min_periods=1).mean()

    #Calculate RSI
    sample_df.loc[:, "RS"] = sample_df["Avg Gain"] / sample_df["Avg Loss"]
    sample_df.loc[:, "RSI"] = sample_df["RS"].apply(lambda x: 100-(100/(1 + x)) if x != 0 else 100)



    return sample_df

def get_latest_value(df):
    return df.tail(1)

###
# STEP 3 - CREATE GRAPHS WITH PRICE AND RSI
###

def plot_data(df):
    # Create subplots with different sizes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plotting the first line (Y1) on the top subplot
    ax1.plot(df.index, df['Adj Close'], label='Price', color='blue')

    # Customize the top subplot
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plotting the second line (Y2) as a horizontal line on the bottom subplot
    ax2.plot(df.index, df['RSI'], label='RSI', color='orange')

    # Customize the bottom subplot
    ax2.set_xlabel('RSI')
    ax2.set_ylabel('')
    ax2.legend()

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    return fig, (ax1, ax2)

def export_data(df):
    # Export data to a csv
    df.to_csv("Output.csv")

def animate_plot(df):
    # Sample data generation within a while loop
    num_items = df.shape[0]
    max_iterations = df.shape[1]

    # Initialize an empty list for each item's values
    item_values = [[] for _ in range(num_items)]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Set the x-axis limits to be constant
    ax.set_xlim(0, max_iterations)

    # Set the y-axis limits between -2 and 2
    ax.set_ylim(-2, 2)

    # Initialize x-axis data (constant throughout the plot)
    x_data = np.arange(max_iterations)

    # Plot initial empty lines for each item
    lines = [ax.plot([], [], label=f'Item {i+1}')[0] for i in range(num_items)]
    ax.set_title('Dynamic Line Graph')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')

    # Function to update the plot for each iteration
    def update(frame):
        # Generate values for each item (replace this with your logic)
        values = [2 * np.random.rand() - 1 for _ in range(num_items)]  # Values between -1 and 1

        # Update the values for each item in the list
        for i, line in enumerate(lines):
            item_values[i].append(values[i])
            line.set_data(x_data[:len(item_values[i])], item_values[i])

        return lines

    # Create an animation and assign it to a variable
    ani = animation.FuncAnimation(fig, update, frames=max_iterations, repeat=False)

    # Display the animation
    plt.show()

###
# STEP 4 - BUY/SELL LOOP LOGIC
###

#review whether having these separate functions is necessary
def order_buy(current_price):
    #if not owned, buy_sentiemnt
    return current_price

def order_sell(balance, buy_price, current_price):
    #if owned, sell_sentiment

    #Make 'held' variable False

    ret = current_price / buy_price - 1
    balance = balance * (1 + ret)

    return balance, ret

###
# STEP 5 - While loop ticking at a given rate to simulate real market data. Take given inputs to determine if buy_sentiemnt or sell_sentiment,
#          maybe consider calculating RSI and MA in real time instead of importing with pandas columns
###

class Bot:
    def __init__(self, name, balance, sentiment, held, held_price):
        self.sentiment = sentiment
        self.balance = balance
        self.name = name
        self.held = held
        self.held_price = held_price


def market_loop(price_data, max_time):
    running = True
    time = 0    #time period
    trade_history = {}  #tracks buy_sentiemnt/sell_sentiment orders
    balance = 1000 #assume starting balance in USD
    held = False    #start without the stock

    sentiment = 0   #to be removed later

    while running:
        price = price_data[time]
        
        #to be replaced with NEAT output
        sentiment = np.random.random()

        if held == False and sentiment > 0.9:
            buy_price = order_buy(price)
            trade_history[time] = "Buy"
            held = True

        elif held and sentiment < 0.1:
            balance = order_sell(balance, buy_price, price)
            trade_history[time] = "Sell"
            held = False

        '''
        time.sleep(1) #add delay to approximate real-time data being fed in
        '''
        time += 1
        print("Time: ", time)

        if time == max_time:
            running = False
            print("Final balance: ", balance)
            return trade_history

'''
# STEP 6 - AI portion, set fitting function (tanh), give the RSI, MA, Price as input, receive buy_sentiemnt/sell_sentiment orders as output
            and run the simulation for the current population of bots and sets fitness based on the balance changes
'''

def fitting_function(genomes, config):
    # Global variables
    global gen
    gen += 1

    # Global df for output
    # global df_output

    # Specify the tickers
    tickers = ['AAPL', 'GOOGL', 'BTC-USD']

    # Set the start and end dates for the data
    start_date = '2024-02-22'
    end_date = '2024-02-28'
    interval = '1m'

    #Choose sample
    position = np.random.randint(0,len(tickers))
    sample = tickers[position]

    df_a = get_data(tickers, start_date, end_date, interval)
    df = create_data(df_a, sample)

    # Create list to hold ge genomes, the bot objects, and NN
    nets = []
    bots = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        bot = Bot(genome_id, 1000, 0, False, 0)  
        bots.append(bot)
        # Initialize an empty list for each item's values
        # item_values = [[] for _ in range(len(bots))]

    iteration = 500
    running = True
    while running:
        price = df.iloc[iteration]['Adj Close']
        rsi = df.iloc[iteration]['RSI']
        ma15 = df.iloc[iteration]['MA 15']
        volume = df.iloc[iteration]['Volume']

        for i, bot in enumerate(bots):
            '''
            #to be replaced with NEAT output
            bot.sentiment = np.random.random()
            '''
            # EXAMPLE CODE FOR BOT SENTIMENT
            bot.sentiment = nets[bots.index(bot)].activate((price, rsi, ma15, volume))[0]


            if bot.held == False and bot.sentiment > 0.5:
                buy_price = order_buy(price)
                bot.held = True
                bot.held_price = buy_price

            elif bot.held and bot.sentiment < -0.5:
                bot.balance, ret = order_sell(bot.balance, bot.held_price, price)
                ge[i].fitness += ret
                bot.held = False

        '''
        time.sleep(1) #add delay to approximate real-time data being fed in
        '''
        iteration += 1

        if iteration == 1500:
            print("Size: ", len(bots))
            print(f"Gen {gen} completed.")
            running = False

# Runs the NEAT algorithm to train a nn
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    max_gen = 50
    winner = p.run(fitting_function, max_gen)

    # show final stats
    print('\nBest genome in gen {!s}:\n{!s}'.format(gen, winner))

###
# STEP X - Final main
###

def main():

    ''' TEST FOR MARKET MECHANICS
    for i in range(100):
        print("i: ", i)
        history = market_loop(df['Adj Close'], 1400)

    plot_data(df)
    plt.show()

    #Export if necessary
    #export_data(df)

    '''


if __name__ == "__main__":
    #check if this works
    gen = 0
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
