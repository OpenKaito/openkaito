curl -X POST --location 'http://127.0.0.1:8000/text_embeddings' \
    --header "x-api-key: $YOUR_SN5_API_KEY" \
    --header 'Content-Type: application/json' \
    --data '{
    "texts": [
        "hello",
        "world",
        "SN5"
    ],
    "miner_uid": 0
}'
