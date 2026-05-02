import json

with open("blockchain/ledger.json") as f:
    chain = json.load(f)

valid = True

for i in range(1, len(chain)):
    if chain[i]["previous_hash"] != chain[i-1]["hash"]:
        valid = False

print("Blockchain valid:", valid)