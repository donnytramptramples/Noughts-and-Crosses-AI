# genetic_algorithm.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import TicTacToeAI
from game import TicTacToe

def crossover(selected_models):
    new_population = []
    for _ in range(len(selected_models)):
        parent1, parent2 = np.random.choice(selected_models, size=2, replace=False)
        child = TicTacToeAI()
        child.fc1.weight.data = crossover_weights(parent1.fc1.weight.data, parent2.fc1.weight.data)
        child.fc1.bias.data = crossover_weights(parent1.fc1.bias.data, parent2.fc1.bias.data)
        child.fc2.weight.data = crossover_weights(parent1.fc2.weight.data, parent2.fc2.weight.data)
        child.fc2.bias.data = crossover_weights(parent1.fc2.bias.data, parent2.fc2.bias.data)
        new_population.append(child)
    return new_population

def crossover_weights(weight1, weight2):
    mask = (torch.rand_like(weight1) < 0.5).float()
    return mask * weight1 + (1 - mask) * weight2

def mutate_population(population, mutation_rate):
    for model in population:
        for param in model.parameters():
            if torch.rand(1).item() < mutation_rate:
                param.data += torch.randn_like(param.data) * mutation_rate

def evaluate_model(model, board):
    board_flat = board.flatten().astype(float)
    model_input = torch.tensor([board_flat], dtype=torch.float32)
    with torch.no_grad():
        model_output = model(model_input)
    return model_output.item()

def train_genetic_algorithm_tictactoe(population_size, generations, mutation_rate, save_file="training_data.bin"):
    population = [TicTacToeAI() for _ in range(population_size)]
    optimizer = optim.SGD(population[0].parameters(), lr=0.1)

    training_data = {"generations": [], "best_scores": []}

    for generation in range(generations):
        scores = []

        for model in population:
            score = evaluate_model(model, np.zeros((3, 3), dtype=int))
            scores.append(score)

        training_data["generations"].append(generation)
        training_data["best_scores"].append(max(scores) if scores else None)

        if scores and population and any(scores):  # Check if scores, population, and any(scores) are not empty or None
            selected_indices = np.argsort(scores)[-population_size:]
            selected_models = [population[i] for i in selected_indices]
            new_population = crossover(selected_models)
            mutate_population(new_population, mutation_rate)

            population = new_population

        if generation % 1000 == 0 and scores and any(selected_indices):  # Check if scores and selected_indices are not empty  # Check if scores and selected_indices are not empty
            best_model_idx = np.argmax(scores) if any(scores) else None
            if best_model_idx is not None and best_model_idx < len(population):  # Check if best_model_idx is within the range of the population
                print("Saving checkpoint...")
                torch.save(population[best_model_idx].state_dict(), f"model_checkpoint_{generation}.pt")
                print("Checkpoint saved.")
                # Save the training data when a model checkpoint is saved
                with open(save_file, "wb") as file:
                    torch.save(training_data, file)

        print(f"Generation {generation + 1}, Best Score: {max(scores) if scores else None}")
        print(f"Population size: {len(population)}, Scores size: {len(scores)}, Selected indices size: {len(selected_indices)}")

    with open(save_file, "wb") as file:
        torch.save(training_data, file)
