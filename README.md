# Mahalanobis distance-based client reliability measurement in Federated Learning

This repository contains the code developed during an internship at the **Medical and Environmental Computing Lab at Technische Universität Darmstadt**, under the supervision of **Mirko Konstantin** and **Dr. Anirban Mukhopadhyay**. The project focuses on implementing a **Mahalanobis distance-based client reliability measurement** to detect and exclude malicious or malfunctioning client updates from the aggregation process, thus preserving the global model’s performance.

## Overview

The goal of this project was to enhance the security and robustness of federated learning by developing mechanisms to detect unreliable clients. This system helps safeguard the global model from malicious updates and improves the overall reliability of the federated learning process.

### Key Contributions:

- Implementation of a **Mahalanobis distance-based reliability measurement** to evaluate client updates.
- Detection of malicious and malfunctioning clients by analyzing their updates.
- Prevention of unreliable updates from being aggregated into the global model.

This repository includes all the code and research done during the internship, including:

- Client-side and server-side federated learning code.
- Integration of Mahalanobis distance-based reliability checks.
- Simulated environment for testing with multiple clients and a central server.

## How It Works

1. **Federated Learning**: Multiple clients train models locally on their data and send updates to a central server.
2. **Reliability Check**: The central server uses Mahalanobis distance to evaluate the reliability of client updates based on their distribution. Clients with high Mahalanobis distances (i.e., those whose updates deviate significantly) are considered unreliable and hence excluded from aggregation.
3. **Model Aggregation**: Updates from reliable clients are aggregated into the global model
