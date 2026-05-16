from nfstream import NFStreamer
import pandas as pd

streamer = NFStreamer(source="traffic.pcap", statistical_analysis=True)

flows = []

for flow in streamer:
    flows.append(flow.to_dict())

df = pd.DataFrame(flows)

df.to_csv("flows.csv", index=False)

print("Flow extraction completed!")