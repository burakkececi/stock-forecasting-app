# Stock-Forecasting
Project Overview:
In finance, stock (also capital stock) consist of all the shares by which ownership of a corporation or company
is divided. 1 A stock exchange is an exchange (or bourse) where stockbrokers and traders can buy and sell shares
(equity stock), bonds, and other securities. Many large companies have their stocks listed on a stock exchange. This
makes the stock more liquid and thus more attractive to many investors. 2 Stock prices are dependent on a lot of
external factors therefore there is no clear rules for forecasting prices. But even with their volatile nature forecasting,
with correct price visualization and statistical modeling, helps investors decide on which stocks to invest for better
returns.
In this project a web application, a backend API and ML models will be developed for the purpose of
visualizing and forecasting stocks. All packages hosted on https://pypi.org/ can be used to develop the software
project.
Objectives:
Your team is tasked with;
  - Deciding on a data source (such as Yahoo Finance, Bloomberg, Nasdaq, etc.)
  - Deciding on historical data start date (up to 5 years)
  - Gathering historical data until 1th Of April, 2023 from decided data source for NASDAQ 100
Technology Sector (NDXT) Companies 3.
  - Store relevant data in csv files.
  - Develop Models For Forecasting Prices For 7 Days, 14 Days, 30 Days. (Software project output should have python project only, Jupyter Notebooks implementations not accepted.)
- Develop a backend API for accessing data;
  o Will serve supported ticker list
  o Will serve trained model outputs
  o Will serve historical prices
  o Will serve additional services required by web application frontend.
- Develop a web application for forecast visualization;
  o Web Application will be developed using Dash. (a python visualization framework)
  o Will provide a drop down menu for selecting ticker (such as NVDA)
  o Will provide a drop down menu for selecting forecast range (7 Days, 14 Days, 1 Month)
  o Will provide start date selection menu for historical data start date
  o Will read trained model outputs, historical prices then show and visualize relevant price history and forecasted price charts.
  o Will provide the ability to compare 2 tickers
- Present your solution and reasoning behind your implemented ML model configurations


## To start the project
The program runs at default port which is https://localhost:8050

```
python display.py
```