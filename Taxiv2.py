import numpy as np
import gymnasium as gym
import json
import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt


@dataclass
class Individual:
    """Klasa reprezentująca osobnika w populacji"""
    id: int
    genotype: List[int]  # Bezpośrednia polityka: akcja dla każdego stanu
    fitness: float = 0.0
    raw_reward: float = 0.0
    success_rate: float = 0.0
    avg_steps: float = 0.0

class TaxiGeneticAlgorithm:
    """Ulepszona wersja algorytmu genetycznego dla Taxi-v3"""
    
    def __init__(self, population_size: int = 100, num_generations: int = 50, 
                 mutation_rate: float = 0.15, crossover_rate: float = 0.85,
                 elite_size: int = 10, seed: int = 42):
        """Inicjalizacja algorytmu genetycznego"""
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed = seed
        
        # Inicjalizacja generatora liczb losowych
        self.rng = np.random.default_rng(seed=seed)
        
        # Parametry środowiska
        self.env = gym.make('Taxi-v3')
        self.state_space = self.env.observation_space.n  # 500 stanów
        self.action_space = self.env.action_space.n      # 6 akcji
        
        # Rozmiar genotypu = liczba stanów
        self.genotype_size = self.state_space
        
        # Lokalizacje w Taxi-v3: R(0,0), G(0,4), Y(4,0), B(4,3)
        self.locations = {
            0: (0, 0),  # Red
            1: (0, 4),  # Green  
            2: (4, 0),  # Yellow
            3: (4, 3)   # Blue
        }
        
        # Mapa środowiska Taxi-v3 (ściany)
        self.taxi_map = [
            "+---------+",
            "|R: | : :G|",
            "| : | : : |",
            "| : : : : |",
            "| | : | : |",
            "|Y: | :B: |",
            "+---------+"
        ]
        
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Analiza dostępnych ruchów dla każdej pozycji
        self.valid_moves = self._compute_valid_moves()
        
    def _compute_valid_moves(self) -> Dict[Tuple[int, int], List[int]]:
        """Oblicza dostępne ruchy dla każdej pozycji na mapie"""
        valid_moves = {}
        
        for row in range(5):
            for col in range(5):
                moves = []
                
                # Sprawdź każdy kierunek
                # Południe (0)
                if row < 4:
                    moves.append(0)
                
                # Północ (1) 
                if row > 0:
                    moves.append(1)
                
                # Wschód (2)
                if col < 4:
                    # Sprawdź czy nie ma pionowej ściany
                    if not ((row == 0 and col == 1) or 
                           (row == 1 and col == 1) or
                           (row == 3 and col == 2) or
                           (row == 4 and col == 2)):
                        moves.append(2)
                
                # Zachód (3)
                if col > 0:
                    # Sprawdź czy nie ma pionowej ściany
                    if not ((row == 0 and col == 2) or 
                           (row == 1 and col == 2) or
                           (row == 3 and col == 3) or
                           (row == 4 and col == 3)):
                        moves.append(3)
                
                valid_moves[(row, col)] = moves
        
        return valid_moves
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Oblicza odległość Manhattan między dwoma pozycjami"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_best_move(self, taxi_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> int:
        """Zwraca najlepszy dostępny ruch w kierunku celu"""
        taxi_row, taxi_col = taxi_pos
        target_row, target_col = target_pos
        
        # Pobierz dostępne ruchy dla aktualnej pozycji
        available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
        
        if not available_moves:
            return self.rng.choice([0, 1, 2, 3])
        
        # Preferowane kierunki do celu
        preferred_moves = []
        
        if taxi_row < target_row and 0 in available_moves:  # Południe
            preferred_moves.append(0)
        if taxi_row > target_row and 1 in available_moves:  # Północ
            preferred_moves.append(1)
        if taxi_col < target_col and 2 in available_moves:  # Wschód
            preferred_moves.append(2)
        if taxi_col > target_col and 3 in available_moves:  # Zachód
            preferred_moves.append(3)
        
        if preferred_moves:
            return self.rng.choice(preferred_moves)
        else:
            return self.rng.choice(available_moves)
    
    def create_heuristic_individual(self, individual_id: int, intelligence_level: float = 0.9) -> Individual:
        """Tworzy osobnika z heurystyką opartą na logice"""
        genotype = []
        
        for state in range(self.genotype_size):
            taxi_row, taxi_col, passenger_loc, destination = self.env.unwrapped.decode(state)
            taxi_pos = (taxi_row, taxi_col)
            
            # Heurystyka w zależności od sytuacji
            if passenger_loc < 4:  # Pasażer czeka na stacji
                passenger_pos = self.locations[passenger_loc]
                
                if taxi_pos == passenger_pos:  # Jesteśmy przy pasażerze
                    if self.rng.random() < intelligence_level:
                        action = 4  # Podnieś pasażera
                    else:
                        action = self.rng.integers(0, self.action_space)
                else:  # Idź po pasażera
                    if self.rng.random() < intelligence_level:
                        action = self.get_best_move(taxi_pos, passenger_pos)
                    else:
                        available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
                        action = self.rng.choice(available_moves)
                        
            elif passenger_loc == 4:  # Pasażer w taksówce
                destination_pos = self.locations[destination]
                
                if taxi_pos == destination_pos:  # Jesteśmy w celu
                    if self.rng.random() < intelligence_level:
                        action = 5  # Zostaw pasażera
                    else:
                        action = self.rng.integers(0, self.action_space)
                else:  # Idź do celu z pasażerem
                    if self.rng.random() < intelligence_level:
                        action = self.get_best_move(taxi_pos, destination_pos)
                    else:
                        available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
                        action = self.rng.choice(available_moves)
            else:
                # Nieprawidłowy stan - losowa akcja ruchu
                available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
                action = self.rng.choice(available_moves)
            
            genotype.append(action)
        
        return Individual(id=individual_id, genotype=genotype)
    
    def create_random_individual(self, individual_id: int) -> Individual:
        """Tworzy losowego osobnika z preferencją dla ruchów"""
        genotype = []
        for state in range(self.genotype_size):
            taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
            taxi_pos = (taxi_row, taxi_col)
            
            # 70% szans na ruch, 15% na pickup, 15% na dropoff
            if self.rng.random() < 0.7:
                available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
                action = self.rng.choice(available_moves)
            else:
                action = self.rng.choice([4, 5])  # pickup lub dropoff
            
            genotype.append(action)
        
        return Individual(id=individual_id, genotype=genotype)
    
    def initialize_population(self):
        """Inicjalizuje populację z różnymi strategiami"""
        self.population = []
        
        # 55% bardzo inteligentnych (0.85-0.95)
        very_smart_count = int(0.55 * self.population_size)
        for i in range(very_smart_count):
            intelligence = 0.85 + self.rng.random() * 0.1
            individual = self.create_heuristic_individual(i, intelligence)
            self.population.append(individual)
        
        # 30% średnio inteligentnych (0.65-0.85)  
        smart_count = int(0.3 * self.population_size)
        for i in range(very_smart_count, very_smart_count + smart_count):
            intelligence = 0.65 + self.rng.random() * 0.2
            individual = self.create_heuristic_individual(i, intelligence)
            self.population.append(individual)
        
        # 10% słabo inteligentnych (0.4-0.65)
        weak_smart_count = int(0.10 * self.population_size)
        for i in range(very_smart_count + smart_count, very_smart_count + smart_count + weak_smart_count):
            intelligence = 0.4 + self.rng.random() * 0.25
            individual = self.create_heuristic_individual(i, intelligence)
            self.population.append(individual)
        
        # 5% całkowicie losowych
        for i in range(very_smart_count + smart_count + weak_smart_count, self.population_size):
            individual = self.create_random_individual(i)
            self.population.append(individual)
            
        print(f"Zainicjalizowano populację {self.population_size} osobników:")
        print(f"- Bardzo inteligentnych (85-95%): {very_smart_count}")
        print(f"- Średnio inteligentnych (65-85%): {smart_count}")
        print(f"- Słabo inteligentnych (40-65%): {weak_smart_count}")
        print(f"- Losowych: {self.population_size - very_smart_count - smart_count - weak_smart_count}")
    
    def evaluate_individual(self, individual: Individual, episodes: int = 20) -> float:
        """Ulepszona ocena osobnika z wieloma metrykami"""
        total_rewards = []
        successful_episodes = 0
        total_steps = []
        pickup_successes = 0
        dropoff_successes = 0
        illegal_actions = 0
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            max_steps = 20
            step = 0
            episode_pickups = 0
            episode_dropoffs = 0
            episode_illegal = 0
            
            while not done and step < max_steps:
                action = individual.genotype[state]
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                
                # Analiza akcji
                if reward == 20:  # Sukces - dotarcie do celu
                    episode_dropoffs += 1
                elif reward == -10:  # Illegalna akcja
                    episode_illegal += 1
                elif reward == -1:  # Normalna akcja
                    if action == 4:  # Próba pickup
                        taxi_row, taxi_col, passenger_loc, _ = self.env.unwrapped.decode(state)
                        if passenger_loc < 4:
                            passenger_pos = self.locations[passenger_loc]
                            if (taxi_row, taxi_col) == passenger_pos:
                                episode_pickups += 1
                
                state = next_state
                step += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(step)
            pickup_successes += episode_pickups
            dropoff_successes += episode_dropoffs
            illegal_actions += episode_illegal
            
            if episode_reward > 0:
                successful_episodes += 1
        
        # Metryki
        avg_reward = np.mean(total_rewards)
        success_rate = successful_episodes / episodes
        avg_steps = np.mean(total_steps)
        pickup_rate = pickup_successes / episodes
        dropoff_rate = dropoff_successes / episodes
        illegal_rate = illegal_actions / episodes
        
        # Zapisz dodatkowe metryki
        individual.raw_reward = avg_reward
        individual.success_rate = success_rate
        individual.avg_steps = avg_steps
        
        # Ulepszona funkcja fitness
        fitness = avg_reward  # Bazowa nagroda
        
        # Duży bonus za sukces
        fitness += success_rate * 500
        
        # Bonus za prawidłowe akcje
        fitness += pickup_rate * 200
        fitness += dropoff_rate * 300
        
        # Kara za nielegalne akcje
        fitness -= illegal_rate * 100
        
        # Bonus za efektywność (mniej kroków gdy sukces)
        if success_rate > 0:
            efficiency_bonus = max(0, (150 - avg_steps) * 0.2)
            fitness += efficiency_bonus
               
        return max(fitness, avg_reward)  # Fitness nie może być gorszy niż średnia nagroda
    
    def evaluate_population(self):
        """Ocenia całą populację"""
        for i, individual in enumerate(self.population):
            individual.fitness = self.evaluate_individual(individual)
        
        # Sortuj populację według fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Zapisz historię
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        print(f"Najlepszy fitness: {best_fitness:.2f}")
        print(f"Średni fitness: {avg_fitness:.2f}")
    
    def tournament_selection(self, tournament_size: int = 20) -> Individual:
        """Selekcja turniejowa"""
        tournament_size = min(tournament_size, len(self.population))
        tournament_indices = self.rng.choice(len(self.population), tournament_size, replace=False)
        tournament = [self.population[i] for i in tournament_indices]
        return max(tournament, key=lambda x: x.fitness)
    
    def uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Krzyżowanie jednostajne z kontekstem"""
        if self.rng.random() > self.crossover_rate:
            child1 = Individual(id=-1, genotype=parent1.genotype.copy())
            child2 = Individual(id=-1, genotype=parent2.genotype.copy())
            return child1, child2
        
        child1_genotype = []
        child2_genotype = []
        
        for i in range(len(parent1.genotype)):
            if self.rng.random() < 0.5:
                child1_genotype.append(parent1.genotype[i])
                child2_genotype.append(parent2.genotype[i])
            else:
                child1_genotype.append(parent2.genotype[i])
                child2_genotype.append(parent1.genotype[i])
        
        child1 = Individual(id=-1, genotype=child1_genotype)
        child2 = Individual(id=-1, genotype=child2_genotype)
        
        return child1, child2
    
    def smart_mutate(self, individual: Individual):
        """Inteligentna mutacja z kontekstem"""
        # Adaptacyjna stopa mutacji - maleje z czasem
        adaptive_rate = self.mutation_rate * (1.0 - self.generation / self.num_generations)
        
        for state in range(len(individual.genotype)):
            if self.rng.random() < adaptive_rate:
                taxi_row, taxi_col, passenger_loc, destination = self.env.unwrapped.decode(state)
                taxi_pos = (taxi_row, taxi_col)
                
                # Kontekstowa mutacja
                if passenger_loc < 4:  # Pasażer czeka
                    passenger_pos = self.locations[passenger_loc]
                    if taxi_pos == passenger_pos:
                        # Przy pasażerze - preferuj pickup
                        new_action = self.rng.choice([4, 0, 1, 2, 3], p=[0.7, 0.075, 0.075, 0.075, 0.075])
                    else:
                        # Nie przy pasażerze - preferuj sensowny ruch
                        available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
                        best_move = self.get_best_move(taxi_pos, passenger_pos)
                        if best_move in available_moves:
                            # 60% szans na najlepszy ruch, 40% na losowy z dostępnych
                            if self.rng.random() < 0.6:
                                new_action = best_move
                            else:
                                new_action = self.rng.choice(available_moves)
                        else:
                            new_action = self.rng.choice(available_moves)
                            
                elif passenger_loc == 4:  # Pasażer w taksówce
                    destination_pos = self.locations[destination]
                    if taxi_pos == destination_pos:
                        # W celu - preferuj dropoff
                        new_action = self.rng.choice([5, 0, 1, 2, 3], p=[0.7, 0.075, 0.075, 0.075, 0.075])
                    else:
                        # Nie w celu - preferuj sensowny ruch
                        available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
                        best_move = self.get_best_move(taxi_pos, destination_pos)
                        if best_move in available_moves:
                            if self.rng.random() < 0.6:
                                new_action = best_move
                            else:
                                new_action = self.rng.choice(available_moves)
                        else:
                            new_action = self.rng.choice(available_moves)
                else:
                    # Fallback
                    available_moves = self.valid_moves.get(taxi_pos, [0, 1, 2, 3])
                    new_action = self.rng.choice(available_moves)
                
                individual.genotype[state] = new_action
    
    def create_next_generation(self):
        """Tworzy następną generację"""
        print(f"Tworzenie następnej generacji...")
        
        next_generation = []
        
        # Elityzm - najlepsi przechodzą bez zmian
        for i in range(self.elite_size):
            elite = Individual(id=i, genotype=self.population[i].genotype.copy())
            elite.fitness = self.population[i].fitness
            elite.raw_reward = self.population[i].raw_reward
            elite.success_rate = self.population[i].success_rate
            elite.avg_steps = self.population[i].avg_steps
            next_generation.append(elite)
        
        # Reszta przez reprodukcję
        while len(next_generation) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.uniform_crossover(parent1, parent2)
            
            self.smart_mutate(child1)
            self.smart_mutate(child2)
            
            next_generation.extend([child1, child2])
        
        # Przytnij do odpowiedniego rozmiaru
        next_generation = next_generation[:self.population_size]
        for i, individual in enumerate(next_generation):
            individual.id = i
        
        self.population = next_generation
    
    def print_generation_stats(self):
        """Wyświetla szczegółowe statystyki generacji"""
        fitnesses = [ind.fitness for ind in self.population]
        success_rates = [ind.success_rate for ind in self.population if hasattr(ind, 'success_rate')]
        
        best_fitness = max(fitnesses)
        avg_fitness = np.mean(fitnesses)
        worst_fitness = min(fitnesses)
        
        print(f"\n{'='*60}")
        print(f"GENERACJA {self.generation}")
        print(f"{'='*60}")
        print(f"Najlepszy fitness: {best_fitness:.2f}")
        print(f"Średni fitness: {avg_fitness:.2f}")
        print(f"Najgorszy fitness: {worst_fitness:.2f}")
        
        if success_rates:
            avg_success = np.mean(success_rates)
            best_success = max(success_rates)
            successful_individuals = sum(1 for sr in success_rates if sr > 0)
            
            print(f"Najlepszy sukces: {best_success:.1%}")
            print(f"Średni sukces: {avg_success:.1%}")
            print(f"Osobników z sukcesem: {successful_individuals}/{len(success_rates)}")
        
        # Rozkład fitness
        excellent = sum(1 for f in fitnesses if f > 50)
        good = sum(1 for f in fitnesses if f > 0)
        poor = sum(1 for f in fitnesses if f <= 0)
        
        print(f"Rozkład fitness:")
        print(f"  Doskonały (>50): {excellent}")
        print(f"  Dobry (>0): {good}")
        print(f"  Słaby (<=0): {poor}")
    
    def plot_fitness_progress(self):
        """Rysuje wykres postępu uczenia"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_history, label='Najlepszy fitness', color='green', linewidth=2)
        plt.plot(self.avg_fitness_history, label='Średni fitness', color='blue', linestyle='--')
        plt.title('Postęp uczenia - fitness w czasie')
        plt.xlabel('Generacja')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Uruchamia algorytm genetyczny"""
        print("="*70)
        print("ULEPSONY ALGORYTM GENETYCZNY DLA TAXI-V3")
        print("="*70)
        print(f"Parametry:")
        print(f"- Wielkość populacji: {self.population_size}")
        print(f"- Liczba generacji: {self.num_generations}")
        print(f"- Stopa mutacji: {self.mutation_rate}")
        print(f"- Stopa krzyżowania: {self.crossover_rate}")
        print(f"- Rozmiar elity: {self.elite_size}")
        
        self.initialize_population()
        
        for generation in range(self.num_generations):
            self.generation = generation
            
            self.evaluate_population()
            self.print_generation_stats()

            self.save_generation_to_file()
            
            if generation < self.num_generations - 1:
                self.create_next_generation()
        
        print(f"\n{'='*70}")
        print(f"ALGORYTM ZAKOŃCZONY!")
        print(f"{'='*70}")
        
        best_individual = max(self.population, key=lambda x: x.fitness)
        print(f"Najlepszy fitness: {best_individual.fitness:.2f}")
        print(f"Najlepszy sukces: {best_individual.success_rate:.1%}")
        print(f"Średnie kroki: {best_individual.avg_steps:.1f}")

        self.plot_fitness_progress()
        
        return best_individual

    def save_generation_to_file(self, filename: str = "genetic_progress.json"):
        """Zapisuje dane generacji do pliku JSON"""
        data = {
            "nr_generacji": int(self.generation),  # Konwersja na int
            "n_populacji": int(len(self.population)),  # Konwersja na int
            "osobnicy": [
                {
                    "id": int(individual.id),  # Konwersja na int
                    "genotyp": [int(g) for g in individual.genotype],  # Konwersja każdego elementu genotypu na int
                    "fitness": float(individual.fitness)  # Konwersja na float
                }
                for individual in self.population
            ]
        }
        
        # Zapisz dane do pliku z niestandardowymi separatorami
        with open(filename, "w") as file:
            json.dump(data, file, indent=4, separators=(",", ": "))

def demonstrate_best_model(best_individual, num_episodes: int = 3):
    """Demonstracja najlepszego modelu"""
    print(f"\n{'='*60}")
    print(f"DEMONSTRACJA NAJLEPSZEGO MODELU")
    print(f"{'='*60}")
    
    env = gym.make('Taxi-v3', render_mode='human')
    action_names = ["Południe↓", "Północ↑", "Wschód→", "Zachód←", "Podnieś", "Zostaw"]
    
    total_rewards = []
    total_steps = []
    successful_episodes = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n--- EPIZOD {episode + 1} ---")
        env.render()
        
        while not done and step < 20:
            action = best_individual.genotype[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            env.render()  # Renderuj po każdym kroku
            
            if reward == 20:
                print(f"  ✓ SUKCES w kroku {step}! (+20)")
                time.sleep(2)  # Dłuższa pauza przy sukcesie
                break
            elif reward == -10:
                print(f"  ✗ Błąd w kroku {step}: {action_names[action]} (-10)")
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_steps.append(step)
        
        if episode_reward > 0:
            successful_episodes += 1
            status = "SUKCES ✓"
        else:
            status = "PORAŻKA ✗"
        
        print(f"Wynik: {status}, Nagroda: {episode_reward}, Kroki: {step}")
        time.sleep(1)  # Pauza między epizodami
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"PODSUMOWANIE")
    print(f"{'='*60}")
    print(f"Udane epizody: {successful_episodes}/{num_episodes}")
    print(f"Średnia nagroda: {np.mean(total_rewards):.2f}")
    print(f"Średnie kroki: {np.mean(total_steps):.1f}")


def main():
    """Funkcja główna"""
    ga = TaxiGeneticAlgorithm(
        population_size=80,      # Zwiększona populacja
        num_generations=1000,      # Więcej generacji
        mutation_rate=0.06,      # Wyższa stopa mutacji
        crossover_rate=0.85,     # Wyższa stopa krzyżowania
        elite_size=15,            # Więcej elit
        seed=42
    )
    
    best_individual = ga.run()
    
    print(f"\n{'='*70}")
    print(f"NAJLEPSZY OSOBNIK")
    print(f"{'='*70}")
    print(f"Fitness: {best_individual.fitness:.2f}")
    print(f"Sukces: {best_individual.success_rate:.1%}")
    print(f"Średnie kroki: {best_individual.avg_steps:.1f}")
    
    demonstrate_best_model(best_individual, num_episodes=20)

if __name__ == "__main__":
    main()