import gymnasium as gym
import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt


class GeneticTaxiAgent:
    def __init__(self, seed: int = 42):
        """
        Algorytm genetyczny dla środowiska Taxi-v3

        Args:
            seed: Ziarno losowości dla reprodukowalności wyników
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Parametry środowiska Taxi-v3
        self.env = gym.make("Taxi-v3", render_mode=None)
        self.state_size = self.env.observation_space.n  # 500 stanów
        self.action_size = self.env.action_space.n  # 6 akcji

        # Parametry algorytmu genetycznego
        self.population_size = 50
        self.generations = 150
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8
        self.elite_size = 8

        # Historia wyników
        self.fitness_history = []
        self.best_fitness_history = []

    def create_individual(self) -> np.ndarray:
        """Tworzy losowego osobnika (tablicę Q-values)"""
        return np.random.uniform(-1, 1, (self.state_size, self.action_size))

    def create_population(self) -> List[np.ndarray]:
        """Tworzy populację początkową"""
        return [self.create_individual() for _ in range(self.population_size)]

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Oblicza odległość Manhattan między dwoma pozycjami"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def decode_state(self, state: int) -> Tuple[int, int, int, int]:
        """
        Dekoduje stan Taxi-v3 na komponenty:
        - taxi_row, taxi_col: pozycja taksówki
        - passenger_location: lokalizacja pasażera (0-3 dla lokacji, 4 gdy w taksówce)
        - destination: cel podróży (0-3)
        """
        # Lokalizacje w środowisku Taxi-v3
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        destination = state % 4
        state //= 4
        passenger_location = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5

        return taxi_row, taxi_col, passenger_location, destination

    def calculate_distance_reward(self, state: int) -> float:
        """Oblicza nagrodę na podstawie odległości Manhattan"""
        taxi_row, taxi_col, passenger_loc, destination = self.decode_state(state)

        # Lokalizacje w środowisku Taxi-v3
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        taxi_pos = (taxi_row, taxi_col)

        if passenger_loc < 4:  # Pasażer nie jest w taksówce
            # Nagroda za bliskość do pasażera
            passenger_pos = locs[passenger_loc]
            distance_to_passenger = self.manhattan_distance(taxi_pos, passenger_pos)
            return -distance_to_passenger * 0.1
        else:  # Pasażer jest w taksówce
            # Nagroda za bliskość do celu
            dest_pos = locs[destination]
            distance_to_dest = self.manhattan_distance(taxi_pos, dest_pos)
            return -distance_to_dest * 0.1

    def evaluate_individual(self, individual: np.ndarray, episodes: int = 10) -> float:
        """Ocenia osobnika przez uruchomienie kilku epizodów"""
        total_reward = 0

        for _ in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            max_steps = 200

            while steps < max_steps:
                # Wybierz akcję na podstawie Q-table osobnika
                action = np.argmax(individual[state])
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Dodaj nagrodę za odległość Manhattan
                distance_reward = self.calculate_distance_reward(state)
                total_episode_reward = reward + distance_reward

                episode_reward += total_episode_reward
                state = next_state
                steps += 1

                if terminated or truncated:
                    break

            total_reward += episode_reward

        return total_reward / episodes

    def selection(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """Selekcja turniejowa"""
        selected = []
        tournament_size = 3

        for _ in range(self.population_size - self.elite_size):
            # Wybierz losowych kandydatów do turnieju
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            # Wybierz najlepszego z turnieju
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())

        return selected

    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Krzyżowanie jednopunktowe"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Flatten arrays for easier crossover
        flat_parent1 = parent1.flatten()
        flat_parent2 = parent2.flatten()

        # Wybierz punkt krzyżowania
        crossover_point = random.randint(1, len(flat_parent1) - 1)

        # Twórz potomków
        child1_flat = np.concatenate(
            [flat_parent1[:crossover_point], flat_parent2[crossover_point:]]
        )
        child2_flat = np.concatenate(
            [flat_parent2[:crossover_point], flat_parent1[crossover_point:]]
        )

        # Reshape back to original shape
        child1 = child1_flat.reshape(parent1.shape)
        child2 = child2_flat.reshape(parent2.shape)

        return child1, child2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutacja gaussowska"""
        mutated = individual.copy()

        for i in range(individual.shape[0]):
            for j in range(individual.shape[1]):
                if random.random() < self.mutation_rate:
                    # Dodaj szum gaussowski
                    mutated[i, j] += np.random.normal(0, 0.1)
                    # Ogranicz wartości do zakresu
                    mutated[i, j] = np.clip(mutated[i, j], -2, 2)

        return mutated

    def evolve(self) -> np.ndarray:
        """Główna pętla algorytmu genetycznego"""
        print(f"Rozpoczynam ewolucję z ziarnem losowości: {self.seed}")
        print(
            f"Parametry: populacja={self.population_size}, generacje={self.generations}"
        )
        print("-" * 60)

        # Twórz populację początkową
        population = self.create_population()

        for generation in range(self.generations):
            # Ocena populacji
            fitness_scores = []
            for i, individual in enumerate(population):
                fitness = self.evaluate_individual(individual)
                fitness_scores.append(fitness)

            # Zapisz statystyki
            avg_fitness = np.mean(fitness_scores)
            best_fitness = np.max(fitness_scores)
            self.fitness_history.append(avg_fitness)
            self.best_fitness_history.append(best_fitness)

            print(
                f"Generacja {generation+1:3d}: "
                f"Najlepszy: {best_fitness:8.2f}, "
                f"Średni: {avg_fitness:8.2f}"
            )

            # Elitaryzm - zachowaj najlepszych
            elite_indices = np.argsort(fitness_scores)[-self.elite_size :]
            elite = [population[i].copy() for i in elite_indices]

            # Selekcja
            selected = self.selection(population, fitness_scores)

            # Krzyżowanie i mutacja
            new_population = elite.copy()

            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i + 1]
                    child1, child2 = self.crossover(parent1, parent2)

                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    new_population.extend([child1, child2])

            # Upewnij się, że populacja ma właściwy rozmiar
            population = new_population[: self.population_size]

        # Zwróć najlepszego osobnika
        final_fitness = [
            self.evaluate_individual(ind, episodes=20) for ind in population
        ]
        best_idx = np.argmax(final_fitness)
        best_individual = population[best_idx]

        print(f"\nEwolucja zakończona!")
        print(f"Najlepsza końcowa fitness: {final_fitness[best_idx]:.2f}")

        return best_individual

    def test_agent(self, agent: np.ndarray, episodes: int = 10, render: bool = False):
        """Testuje wytrenowanego agenta"""
        if render:
            test_env = gym.make("Taxi-v3", render_mode="human")
        else:
            test_env = self.env

        total_rewards = []
        successful_deliveries = 0

        for episode in range(episodes):
            state, _ = test_env.reset()
            episode_reward = 0
            steps = 0
            max_steps = 200

            while steps < max_steps:
                action = np.argmax(agent[state])
                state, reward, terminated, truncated, _ = test_env.step(action)
                episode_reward += reward
                steps += 1

                if terminated:
                    if reward == 20:  # Successful delivery
                        successful_deliveries += 1
                    break
                elif truncated:
                    break

            total_rewards.append(episode_reward)
            print(f"Epizod {episode+1}: Nagroda = {episode_reward}, Kroki = {steps}")

        if render:
            test_env.close()

        avg_reward = np.mean(total_rewards)
        success_rate = successful_deliveries / episodes * 100

        print(f"\nWyniki testowania:")
        print(f"Średnia nagroda: {avg_reward:.2f}")
        print(f"Wskaźnik sukcesu: {success_rate:.1f}%")
        print(f"Udane dostawy: {successful_deliveries}/{episodes}")

        return avg_reward, success_rate

    def plot_evolution(self):
        """Wykres postępu ewolucji"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.fitness_history, label="Średnia fitness", alpha=0.7)
        plt.plot(self.best_fitness_history, label="Najlepsza fitness", alpha=0.9)
        plt.xlabel("Generacja")
        plt.ylabel("Fitness")
        plt.title("Postęp ewolucji")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Wykres poprawy w czasie
        improvement = np.array(self.best_fitness_history) - self.best_fitness_history[0]
        plt.plot(improvement)
        plt.xlabel("Generacja")
        plt.ylabel("Poprawa fitness")
        plt.title("Poprawa względem początku")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Główna funkcja demonstracyjna"""
    # Ustaw ziarno dla reprodukowalności
    SEED = 42

    # Twórz i uruchom algorytm genetyczny
    ga_agent = GeneticTaxiAgent(seed=SEED)

    print("=== ALGORYTM GENETYCZNY DLA TAXI-V3 ===")
    print("Użyte techniki:")
    print("- Odległość Manhattan dla funkcji fitness")
    print("- Selekcja turniejowa")
    print("- Krzyżowanie jednopunktowe")
    print("- Mutacja gaussowska")
    print("- Elitaryzm")
    print("=" * 50)

    # Ewolucja
    best_agent = ga_agent.evolve()

    # Testowanie
    print("\n" + "=" * 50)
    print("TESTOWANIE NAJLEPSZEGO AGENTA")
    print("=" * 50)
    avg_reward, success_rate = ga_agent.test_agent(best_agent, episodes=20)

    # Wykres ewolucji
    ga_agent.plot_evolution()

    # Zapisz najlepszego agenta
    np.save(f"best_taxi_agent_seed_{SEED}.npy", best_agent)
    print(f"\nNajlepszy agent zapisany jako 'best_taxi_agent_seed_{SEED}.npy'")


if __name__ == "__main__":
    main()
