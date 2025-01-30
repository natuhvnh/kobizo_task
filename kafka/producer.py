from confluent_kafka import Producer
import time
import json

conf = {"bootstrap.servers": "localhost:9092"}
producer = Producer(conf)


def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")


topic = "transactions"
data = [
    {
        "transaction id": "0xf04403e162a8a988780de9a0416f8b87633e71435f03785f1c0c1291a6eb330b",
        "time stamp": "Jul-31-2024 12:53:37 PM UTC",
        "from": "0x000000000231c53e9dCbD5Ee410f065FBc170c29",
        "to": "0x00000000041d945c46E073F0048cEf510D148dEA",
        "value": "$10.71",
        "method called": "buy",
        "token price": "$2.38",
        "liquidity": "$1,153,212.96",
        "market cap": "$3,234,533.45",
    }
]
for i in range(len(data)):
    message = data[i]
    producer.produce(topic, key=str(i), value=json.dumps(message), callback=delivery_report)
    producer.poll(0)
    time.sleep(1)

producer.flush()
