from confluent_kafka import Consumer
import requests
import json

conf = {"bootstrap.servers": "localhost:9092", "group.id": "my-group", "auto.offset.reset": "earliest"}

consumer = Consumer(conf)
consumer.subscribe(["transactions"])


def get_model_output(data):
    url = "http://127.0.0.1:5000/predict"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=json.dumps(data), headers=headers)
    print(response.status_code)
    print(response.json())
    return


try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        data = json.loads(msg.value().decode("utf-8"))
        get_model_output(data)
except KeyboardInterrupt:
    pass
finally:
    consumer.close()
