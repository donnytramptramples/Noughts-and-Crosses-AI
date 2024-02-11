from genetic_algorithm import train_genetic_algorithm_tictactoe

def main():
    train_genetic_algorithm_tictactoe(population_size=1000, generations=10000, mutation_rate=0.01, save_file="training_data.bin")

if __name__ == "__main__":
    main()