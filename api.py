import pandas as pd
import pickle
import json
from flask import Flask, request, jsonify
import utils

app = Flask(__name__)
# Load models
with open("model/large_transaction_model.pkl", "rb") as file:
    large_transaction_model = pickle.load(file)
with open("model/rapid_transaction_model.pkl", "rb") as file:
    rapid_transaction_model = pickle.load(file)
with open("model/fraud_model.pkl", "rb") as file:
    fraud_model = pickle.load(file)
#
def feature_engineering(input_data):
    df = pd.DataFrame([input_data])
    #
    # convert currency column to float
    convert_col = ["value", "token price", "liquidity", "market cap"]
    for c in convert_col:
        df[c] = df[c].astype(str).str.replace("$", "").str.replace(",", "").astype(float)
    #
    df = df.rename(
        columns={
            "time stamp": "time_stamp",
            "value": "txn_value",
            "token price": "token_price",
            "market cap": "market_cap",
            "method called": "method_called",
        }
    )
    #
    replace_dict = {"buy": 1, "transfer": 2, "swap": 3, "printMoney": 4}
    df["method"] = df["method_called"].replace(replace_dict)
    #
    df["time_stamp"] = df["time_stamp"].str.replace(r"\s+(AM|PM)", "", regex=True)  # inconsitant timestamp
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="%b-%d-%Y %H:%M:%S %Z")
    #
    df["hour"] = df.time_stamp.dt.hour
    df["weekday_number"] = df.time_stamp.dt.weekday
    #
    current_txn = pd.read_parquet("data/most_current_data.parquet")
    df["time_stamp"] = df["time_stamp"].dt.tz_localize(None)
    df["txn_time_gap"] = (
        df["time_stamp"] - current_txn[current_txn.txn_maker_address == df["from"].values[0]]["time_stamp"].values[0]
    ).dt.total_seconds()
    #
    df["txn_value_to_token_price"] = df["txn_value"] * 100 / df["token_price"]
    df["txn_value_to_liquidity"] = df["txn_value"] * 100 / df["liquidity"]
    df["txn_value_to_market_cap"] = df["txn_value"] * 100 / df["market_cap"]
    return df


# Define a route for predictions
@app.route("/predict", methods=["POST", "GET"])
def predict():
    """
    API endpoint to make predictions.
    The client sends a JSON request with input features.
    """
    input_data = json.loads(request.get_json())
    print(input_data)
    input_data = feature_engineering(input_data)
    # large txn inference
    input_feature = [
        "txn_value",
        "method",
        "token_price",
        "liquidity",
        "market_cap",
        "hour",
        "weekday_number",
        "txn_value_to_token_price",
        "txn_value_to_liquidity",
        "txn_value_to_market_cap",
    ]
    large_transaction = large_transaction_model.predict(input_data[input_feature])
    large_transaction_score = large_transaction_model.decision_function(input_data[input_feature])
    # rapid txn inference
    input_feature = ["method", "hour", "weekday_number", "txn_time_gap"]
    rapid_transaction = rapid_transaction_model.predict(input_data[input_feature])
    rapid_transaction_score = rapid_transaction_model.decision_function(input_data[input_feature])
    # fraud txn inference
    input_feature = [
        "txn_value",
        "method",
        "token_price",
        "liquidity",
        "market_cap",
        "hour",
        "weekday_number",
        "txn_value_to_token_price",
        "txn_value_to_liquidity",
        "txn_value_to_market_cap",
        "method",
        "hour",
        "weekday_number",
        "txn_time_gap",
    ]
    fraud_transaction = fraud_model.predict(input_data[input_feature])
    fraud_transaction_score = fraud_model.decision_function(input_data[input_feature])
    # append & update current data => to be added
    # skip this part for demo and debug purposes
    # write predicted data
    input_data["large_transaction"] = large_transaction
    input_data["large_transaction_score"] = large_transaction_score
    input_data["rapid_transaction"] = rapid_transaction
    input_data["rapid_transaction_score"] = rapid_transaction_score
    input_data["fraud_transaction"] = fraud_transaction
    input_data["fraud_transaction_score"] = fraud_transaction_score
    utils.append_model_output(input_data, "data/model_output.parquet")
    # Return the prediction as JSON
    return jsonify(
        {
            "large_transaction": int(large_transaction[0]),
            "large_transaction_score": large_transaction_score[0],
            "rapid_transaction": int(rapid_transaction[0]),
            "rapid_transaction_score": rapid_transaction_score[0],
            "fraud_transaction": int(fraud_transaction[0]),
            "fraud_transaction_score": fraud_transaction_score[0],
        }
    )


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
