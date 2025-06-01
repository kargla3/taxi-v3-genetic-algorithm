import numpy as np
import gymnasium as gym
import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Individual:
    """Klasa reprezentująca osobnika w populacji"""
    id: int
    genotype: List[float]
    fitness: float = 0.0
    q_table: np.ndarray = None

class TaxiGeneticAlgorithm:
    """Algorytm genetyczny dla środowiska Taxi-v3"""
    
    def __init__(self, population_size: int = 50, num_generations: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elite_size: int = 5, seed: int = 42):
        """
        Inicjalizacja algorytmu genetycznego
        
        Args:
            population_size: Rozmiar populacji
            num_generations: Liczba generacji
            mutation_rate: Prawdopodobieństwo mutacji
            crossover_rate: Prawdopodobieństwo krzyżowania
            elite_size: Liczba najlepszych osobników przechodzących do następnej generacji
            seed: Ziarno dla generatora liczb losowych
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed = seed
        
        # Inicjalizacja generatora liczb losowych - TYLKO RAZ!
        self.rng = np.random.default_rng(seed=seed)
        
        # Parametry środowiska
        self.env = gym.make('Taxi-v3')
        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n
        
        # Rozmiar genotypu: [learning_rate, discount_factor, epsilon_start, epsilon_end, epsilon_decay]
        self.genotype_size = 5
        
        # Ograniczenia dla genów
        self.gene_bounds = [
            (0.01, 0.5),   # learning_rate
            (0.8, 0.99),   # discount_factor
            (0.5, 1.0),    # epsilon_start
            (0.01, 0.1),   # epsilon_end
            (0.995, 0.9999) # epsilon_decay
        ]
        
        self.population = []
        self.generation = 0
        self.progress_file = f"taxi_ga_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def create_individual(self, individual_id: int) -> Individual:
        """Tworzy nowego osobnika z losowym genotypem"""
        genotype = []
        for i in range(self.genotype_size):
            min_val, max_val = self.gene_bounds[i]
            gene = self.rng.uniform(min_val, max_val)
            genotype.append(gene)
        
        return Individual(id=individual_id, genotype=genotype)
    
    def initialize_population(self):
        """Inicjalizuje populację"""
        self.population = []
        for i in range(self.population_size):
            individual = self.create_individual(i)
            self.population.append(individual)
    
    def train_q_learning(self, individual: Individual, episodes: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Trenuje Q-learning z parametrami z genotypu osobnika
        
        Returns:
            Tuple[Q-table, średnia nagroda]
        """
        learning_rate, discount_factor, epsilon_start, epsilon_end, epsilon_decay = individual.genotype
        
        # Inicjalizacja Q-table
        q_table = self.rng.uniform(-1, 1, (self.state_space, self.action_space))
        
        epsilon = epsilon_start
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Epsilon-greedy action selection
                if self.rng.random() < epsilon:
                    action = self.rng.integers(0, self.action_space)
                else:
                    action = np.argmax(q_table[state])
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Q-learning update
                best_next_action = np.argmax(q_table[next_state])
                td_target = reward + discount_factor * q_table[next_state][best_next_action]
                td_error = td_target - q_table[state][action]
                q_table[state][action] += learning_rate * td_error
                
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay
        
        avg_reward = np.mean(total_rewards[-100:])  # Średnia z ostatnich 100 epizodów
        return q_table, avg_reward
    
    def evaluate_individual(self, individual: Individual) -> float:
        """Ocenia osobnika przez trening Q-learning"""
        q_table, avg_reward = self.train_q_learning(individual)
        individual.q_table = q_table
        
        # Dodatkowo testujemy wytrenowany model
        test_rewards = []
        for _ in range(100):  # 100 testowych epizodów
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = np.argmax(q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            test_rewards.append(episode_reward)
        
        # Fitness jako średnia nagroda z testowania
        fitness = np.mean(test_rewards)
        return fitness
    
    def evaluate_population(self):
        """Ocenia całą populację"""
        print(f"Ocenianie populacji generacji {self.generation}...")
        for i, individual in enumerate(self.population):
            individual.fitness = self.evaluate_individual(individual)
            print(f"Osobnik {individual.id}: fitness = {individual.fitness:.2f}")
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Selekcja turniejowa"""
        tournament = self.rng.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Krzyżowanie jednopunktowe"""
        if self.rng.random() > self.crossover_rate:
            return parent1, parent2
        
        crossover_point = self.rng.integers(1, self.genotype_size)
        
        child1_genotype = (parent1.genotype[:crossover_point] + 
                          parent2.genotype[crossover_point:])
        child2_genotype = (parent2.genotype[:crossover_point] + 
                          parent1.genotype[crossover_point:])
        
        child1 = Individual(id=-1, genotype=child1_genotype)
        child2 = Individual(id=-1, genotype=child2_genotype)
        
        return child1, child2
    
    def mutate(self, individual: Individual):
        """Mutacja gaussowska"""
        for i in range(len(individual.genotype)):
            if self.rng.random() < self.mutation_rate:
                min_val, max_val = self.gene_bounds[i]
                # Gaussowska mutacja z ograniczeniem do granic
                mutation = self.rng.normal(0, 0.1)
                individual.genotype[i] += mutation
                # Ograniczenie do dozwolonych wartości
                individual.genotype[i] = np.clip(individual.genotype[i], min_val, max_val)
    
    def create_next_generation(self):
        """Tworzy następną generację"""
        # Sortowanie populacji według fitness (malejąco)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Elityzm - najlepsi przechodzą bez zmian
        next_generation = self.population[:self.elite_size]
        
        # Tworzenie reszty populacji
        while len(next_generation) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            self.mutate(child1)
            self.mutate(child2)
            
            next_generation.extend([child1, child2])
        
        # Przycinamy do odpowiedniego rozmiaru i przypisujemy ID
        next_generation = next_generation[:self.population_size]
        for i, individual in enumerate(next_generation):
            individual.id = i
        
        self.population = next_generation
    
    def save_progress(self):
        """Zapisuje postęp do pliku JSON"""
        progress_data = {
            "nr_generacji": self.generation,
            "n_populacji": self.population_size,
            "osobnicy": []
        }
        
        for individual in self.population:
            individual_data = {
                "id": individual.id,
                "genotyp": [round(gene, 6) for gene in individual.genotype],
                "fitness": round(individual.fitness, 2)
            }
            progress_data["osobnicy"].append(individual_data)
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        print(f"Postęp zapisany do pliku: {self.progress_file}")
    
    def print_generation_stats(self):
        """Wyświetla statystyki generacji"""
        fitnesses = [ind.fitness for ind in self.population]
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        worst_fitness = min(fitnesses)
        
        best_individual = max(self.population, key=lambda x: x.fitness)
        
        print(f"\n=== Generacja {self.generation} ===")
        print(f"Najlepszy fitness: {best_fitness:.2f}")
        print(f"Średni fitness: {avg_fitness:.2f}")
        print(f"Najgorszy fitness: {worst_fitness:.2f}")
        print(f"Najlepszy genotyp: {[round(g, 4) for g in best_individual.genotype]}")
        print(f"Parametry najlepszego: LR={best_individual.genotype[0]:.4f}, "
              f"γ={best_individual.genotype[1]:.4f}, "
              f"ε_start={best_individual.genotype[2]:.4f}, "
              f"ε_end={best_individual.genotype[3]:.4f}, "
              f"ε_decay={best_individual.genotype[4]:.6f}")
    
    def run(self):
        """Uruchamia algorytm genetyczny"""
        print("Inicjalizacja algorytmu genetycznego dla Taxi-v3")
        print(f"Parametry: populacja={self.population_size}, generacje={self.num_generations}")
        print(f"Ziarno RNG: {self.seed}")
        
        # Inicjalizacja populacji
        self.initialize_population()
        
        for generation in range(self.num_generations):
            self.generation = generation
            
            # Ocena populacji
            self.evaluate_population()
            
            # Wyświetlenie statystyk
            self.print_generation_stats()
            
            # Zapis postępu
            self.save_progress()
            
            # Tworzenie następnej generacji (oprócz ostatniej)
            if generation < self.num_generations - 1:
                self.create_next_generation()
        
        print(f"\nAlgorytm zakończony!")
        print(f"Najlepszy wynik: {max(ind.fitness for ind in self.population):.2f}")
        
        # Zwracamy najlepszego osobnika
        best_individual = max(self.population, key=lambda x: x.fitness)
        return best_individual

def main():
    """Funkcja główna"""
    # Tworzenie i uruchomienie algorytmu genetycznego
    ga = TaxiGeneticAlgorithm(
        population_size=20,
        num_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=3,
        seed=71
    )
    
    best_individual = ga.run()
    
    print(f"\nNajlepszy znaleziony osobnik:")
    print(f"Genotyp: {[round(g, 4) for g in best_individual.genotype]}")
    print(f"Fitness: {best_individual.fitness:.2f}")
    
    # Demonstracja działania najlepszego modelu
    print(f"\nDemonstracja najlepszego modelu...")
    env = gym.make('Taxi-v3', render_mode='human')
    
    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpizod {episode + 1}:")
        
        while not done and step < 200:
            action = np.argmax(best_individual.q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        
        print(f"Nagroda: {episode_reward}, Kroki: {step}")
    
    env.close()

if __name__ == "__main__":
    main()