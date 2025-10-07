import flwr as fl

def main():
    # Define the strategy (e.g., FedAvg)
    strategy = fl.server.strategy.FedAvg(
        # We can add more parameters here, like minimum number of clients
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # Start the Flower server
    print("Starting Flower server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080", # Listen on all network interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        # For production, you would add SSL certificates here
        # certificates=(...)
    )

if __name__ == "__main__":
    main()