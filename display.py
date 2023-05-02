import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# reading the stock csv file and convert it to data frame
stocks = pd.read_csv("historical_results.csv")
stocks['Date'] = pd.to_datetime(stocks['Date'])

tickers = stocks['Stock'].sort_values().unique()
min_date = stocks['Date'].min()
max_date = (stocks['Date'].max() - pd.DateOffset(days=30)).date()

# reading the forecast csv file and convert it to data frame
forecasts = pd.read_csv("prediction_results.csv")
forecasts['Date'] = pd.to_datetime(forecasts['Date'])

# init Dash app
app = Dash(__name__)
app.title = "Stock Market Insider"

# layout of the Dash app
app.layout = html.Div([
    html.Header(
        [
            html.H1("Stock Market"),
            html.H1("Insider")
        ], className="header"),
    html.Main(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Select stock:"),
                            dcc.Dropdown(options=tickers, value=None, id="stock_code")
                        ], className="input"
                    ),
                    html.Div(
                        [
                            html.P("Compare stock:"),
                            dcc.Dropdown(options=tickers, value=None, id="stock_code_cmp")
                        ], className="input"
                    ),
                    html.Div(
                        [
                            html.P("Select interval:"),
                            # Date range input, default value will be defined from data
                            dcc.DatePickerRange(
                                id='date_range',
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                start_date=min_date,
                                end_date=max_date
                            )
                        ], className="date"
                    ),
                    html.Div(
                        [
                            html.P("Forecast range in days:"),
                            dcc.Dropdown(options=[7, 14, 30], value=7,
                                         id="forecast_day")
                        ], className="input"
                    ),

                ], className="input-box"),

            html.Hr(),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2(id="ticker_name"),
                                            html.P('NasdaqGS. Currency in USD')  # static for now
                                        ], className="title"),
                                    html.Div(
                                        [
                                            html.H3(id="last_price"),
                                            html.P(id="price_ratio")
                                        ], className="price"),
                                ]),

                            html.Div(
                                [
                                    html.H2("Predicted price:"),
                                    html.H2(id="forecast_price"),
                                    html.P(id="forecast_info")
                                ], className="forecast_graph"
                            ),
                        ], className="price_chart"),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2(id="ticker_name2"),
                                            html.P('NasdaqGS. Currency in USD')  # static for now
                                        ], className="title"),
                                    html.Div(
                                        [
                                            html.H3(id="last_price2"),
                                            html.P(id="price_ratio2")
                                        ], className="price"),
                                ]),

                            html.Div(
                                [
                                    html.H2("Predicted price:"),
                                    html.H2(id="forecast_price2"),
                                    html.P(id="forecast_info2")
                                ], className="forecast_graph"
                            ),
                        ], className="price_chart"),
                ], className="price_box"),

            html.Div(
                [
                    html.Div(
                        dcc.Loading(children=[html.Div([], id='stonks-graph', className='graphs')], id='loading1',
                                    type='graph'), className="graph"
                    ),
                ], className='output_container'),

        ], className="container"),
    html.Footer(
        [
            html.Div(
                html.P("Copyright © 2023"),
                className="copyright"
            )
        ])
], className='')


@app.callback(
    Output("ticker_name", "children"),
    Output("ticker_name2", "children"),
    Input("stock_code", "value"),
    Input("stock_code_cmp", "value"),

)
def update_ticker_title(stock_code, stock_code_cmp):
    """
    Updates the header of the ticker information.

    :param stock_code: str
        The name of the ticker.
    :param stock_code_cmp: str
        The name of the ticker you want to compare.
    :return: str
        Displays the selected ticker names.
    """
    return stock_code, stock_code_cmp


@app.callback(
    Output("last_price", "children"),
    Output("last_price2", "children"),
    Input("stock_code", "value"),
    Input("stock_code_cmp", "value"),
    Input("date_range", "end_date")
)
def update_last_price(stock_code, stock_code_cmp, end_date):
    """
    Updates the last price of the ticker.

    :param stock_code: str
        The name of the ticker.
    :param stock_code_cmp: str
        The name of the ticker you want to compare.
    :param end_date: str
        The last day of the given ticker.
    :return: float
        The last price of the given ticker.
    """
    df = stocks.loc[(stocks['Stock'] == stock_code) & (stocks['Date'] == end_date)]
    last_price = 0
    last_price_cmp = 0
    if not df.empty:
        last_price = df['Close'].values[0]
        if stock_code_cmp is not None:
            # Getting the given stock code with given end date
            df_cmp = stocks.loc[(stocks['Stock'] == stock_code_cmp) & (stocks['Date'] == end_date)]
            last_price_cmp = df_cmp['Close'].values[0]

    return "${:.2f}".format(last_price), "${:.2f}".format(last_price_cmp)


def set_price_ratio(stock_code, end_date):
    # Getting the given stock code with given date range
    df = stocks.loc[(stocks['Stock'] == stock_code) & (stocks['Date'] <= end_date)
                    & (stocks['Date'] >= (pd.Timestamp(end_date) - pd.DateOffset(days=int(1))))]
    last_price = 0
    last_price_rt = 0
    color = "black"
    if not df.empty:
        last_price = df['Close'].values[1] - df['Close'].values[0]
        last_price_rt = ((df['Close'].values[1] - df['Close'].values[0]) / df['Close'].values[0]) * 100
        if last_price > 0:
            color = "green"
        else:
            color = "red"

    return last_price, last_price_rt, color


@app.callback(
    Output("price_ratio", "children"),
    Output("price_ratio", "style"),
    Input("stock_code", "value"),
    Input("date_range", "end_date")
)
def update_price_ratio(stock_code, end_date):
    """
    Updates the price change ratio according to the day before of the given ticker.

    :param stock_code: str
        The name of the ticker.
    :param end_date: str
        The last day of the given ticker.
    :return: float
         The price change ratio of the given ticker.
    """

    last_price, last_price_rt, color = set_price_ratio(stock_code, end_date)

    return "{:.2f} ({:.2f}%)".format(last_price, last_price_rt), {'color': color}


@app.callback(
    Output("price_ratio2", "children"),
    Output("price_ratio2", "style"),
    Input("stock_code_cmp", "value"),
    Input("date_range", "end_date")
)
def update_price_ratio_cmp(stock_code_cmp, end_date):
    """
    Updates the price change ratio according to the day before of the given ticker.

    :param stock_code_cmp: str
        The name of the ticker.
    :param end_date: str
        The last day of the given ticker.
    :return: float
         The price change ratio of the given ticker.
    """

    last_price, last_price_rt, color = set_price_ratio(stock_code_cmp, end_date)

    return "{:.2f} ({:.2f}%)".format(last_price, last_price_rt), {'color': color}


def set_forecast_detail(stock_code, end_date, interval):
    forecast = forecasts.query(f"Stock in ['{stock_code}']")

    price = 0
    f_info = ""
    color = "black"
    if not forecast.empty:
        p_end_date = (pd.Timestamp(end_date) + pd.DateOffset(days=int(interval))).date()
        forecast_interval = forecast.loc[
            (forecast['Date'] >= (pd.Timestamp(end_date) - pd.DateOffset(days=2))) & (
                    forecast['Date'] <= pd.Timestamp(p_end_date))]

        last_price = forecast_interval.iloc[0]['Actual']
        f_last_price = forecast_interval.iloc[-1]['Prediction']
        ratio = ((f_last_price - last_price) / f_last_price) * 100

        if ratio > 0:
            color = "green"
        else:
            color = "red"

        price = "${:.2f} ({:.2f}%)".format(f_last_price, ratio)
        f_info = "Offering {} days price targets for {}. The price target is forecast of ${:.2f}. The price " \
                 "target represents a {:.2f}% change from the last price of ${:.2f}." \
            .format(interval, stock_code, f_last_price, ratio, last_price)

    return price, color, f_info


@app.callback(
    Output("forecast_price", "children"),
    Output("forecast_price", "style"),
    Output("forecast_info", "children"),
    Input("stock_code", "value"),
    Input("date_range", "end_date"),
    Input('forecast_day', 'value')
)
def update_forecast_detail(stock_code, end_date, interval):
    """

    Updates the forecast details of the given ticker.

    :param stock_code: str
        The name of the ticker.
    :param end_date: str
        The last day of the given ticker.
    :param interval: int
        The forecast day to be shown.
    :return:
        The calculated forecast price, if the price is down set color red, otherwise the price is up it is green
        or equal than black. Related information about forecast details also is provided.
    """

    price, color, f_info = set_forecast_detail(stock_code, end_date, interval)

    return price, {'color': color}, f_info


@app.callback(
    Output("forecast_price2", "children"),
    Output("forecast_price2", "style"),
    Output("forecast_info2", "children"),
    Input("stock_code_cmp", "value"),
    Input("date_range", "end_date"),
    Input('forecast_day', 'value')
)
def update_forecast_detail(stock_code_cmp, end_date, interval):
    """

    Updates the forecast details of the given ticker.

    :param stock_code_cmp: str
        The name of the ticker.
    :param end_date: str
        The last day of the given ticker.
    :param interval: int
        The forecast day to be shown.
    :return:
        The calculated forecast price, if the price is down set color red, otherwise the price is up it is green
        or equal than black. Related information about forecast details also is provided.
    """
    price, color, f_info = set_forecast_detail(stock_code_cmp, end_date, interval)

    return price, {'color': color}, f_info


@app.callback(
    Output('stonks-graph', 'children'),
    Input('stock_code', 'value'),
    Input('stock_code_cmp', 'value'),
    Input('date_range', 'start_date'),
    Input('date_range', 'end_date'),
    Input('forecast_day', 'value')
)
def update_stock_graph(stock_code, stock_code_cmp, start_date, end_date, interval):
    """
        Displays a graph that contains given ticker's behaviour.
    :param stock_code:
        The name of the ticker.
    :param stock_code_cmp: str
        The name of the ticker you want to compare.
    :param start_date: str
        The first day of the given ticker.
    :param end_date: str
        The last day of the given ticker.
    :param interval: int
        The forecast day to be shown.
    :return:
        Graph that has ticker's behaviour within given dates and forecasted data after within interval.
    """
    # Create the figure and add two scatter traces
    fig = go.Figure()

    # query for gathering related ticker data.
    stock = stocks.query(f"Stock in ['{stock_code}']")
    if not stock.empty:
        p_end_date = (pd.Timestamp(end_date) + pd.DateOffset(days=int(interval))).date()

        stock_interval = stock.loc[
            (stock['Date'] >= pd.Timestamp(start_date)) & (stock['Date'] <= pd.Timestamp(end_date))]
        # draws a line of the given ticker's data
        fig.add_trace(go.Scatter(
            x=stock_interval['Date'],
            y=stock['Close'],
            mode='lines',
            name=f'{stock_code}',
        ))

        forecast = forecasts.query(f"Stock in ['{stock_code}']")
        forecast_interval = forecast.loc[
            (forecast['Date'] >= (pd.Timestamp(end_date) - pd.DateOffset(days=2))) & (
                    forecast['Date'] <= pd.Timestamp(p_end_date))]

        fig.add_trace(go.Scatter(
            x=forecast_interval['Date'],
            y=forecast['Prediction'],
            mode='lines',
            name=f'{stock_code} Prediction',
            line=dict(dash='dash'),
        ))

        fig.add_trace(go.Scatter(
            x=forecast_interval['Date'],
            y=forecast_interval['Actual'],
            mode='lines',
            name=f'{stock_code} Actual',
        ))

    if stock_code_cmp is not None:
        # query for gathering related ticker data.
        stock_cmp = stocks.query(f"Stock in ['{stock_code_cmp}']")
        if not stock_cmp.empty:
            p_end_date = (pd.Timestamp(end_date) + pd.DateOffset(days=int(interval))).date()

            stock_interval = stock_cmp.loc[
                (stock_cmp['Date'] >= pd.Timestamp(start_date)) & (stock_cmp['Date'] <= pd.Timestamp(end_date))]
            # draws a line of the given ticker's data
            fig.add_trace(go.Scatter(
                x=stock_interval['Date'],
                y=stock_cmp['Close'],
                mode='lines',
                name=f'{stock_code_cmp}',
            ))

            forecast_cmp = forecasts.query(f"Stock in ['{stock_code_cmp}']")
            forecast_interval = forecast_cmp.loc[
                (forecast_cmp['Date'] >= (pd.Timestamp(end_date) - pd.DateOffset(days=2))) & (
                        forecast_cmp['Date'] <= pd.Timestamp(p_end_date))]

            fig.add_trace(go.Scatter(
                x=forecast_interval['Date'],
                y=forecast_cmp['Prediction'],
                mode='lines',
                name=f'{stock_code_cmp} Prediction',
                line=dict(dash='dash'),
            ))

            fig.add_trace(go.Scatter(
                x=forecast_interval['Date'],
                y=forecast_interval['Actual'],
                mode='lines',
                name=f'{stock_code_cmp} Actual',
            ))

    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
