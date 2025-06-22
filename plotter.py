import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import re


class FitnessPlotter:
    """Klasa do generowania wykresów fitness na podstawie plików JSON z populacjami"""

    def __init__(self, populations_dir: str = "populations"):
        """
        Inicjalizacja plottera

        Args:
            populations_dir: Ścieżka do katalogu z plikami JSON populacji
        """
        self.populations_dir = populations_dir
        self.data_cache = {}

    def load_population_data(self) -> List[Dict]:
        """Ładuje dane ze wszystkich plików JSON w katalogu"""
        if not os.path.exists(self.populations_dir):
            print(f"Katalog {self.populations_dir} nie istnieje!")
            return []

        # Znajdź wszystkie pliki JSON z populacjami
        pattern = os.path.join(self.populations_dir, "population_gen_*.json")
        files = glob.glob(pattern)

        if not files:
            print(f"Nie znaleziono plików z populacjami w {self.populations_dir}")
            return []

        # Sortuj pliki według numeru generacji
        files.sort(
            key=lambda x: int(re.search(r"population_gen_(\d+)\.json", x).group(1))
        )

        populations = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    populations.append(data)
            except Exception as e:
                print(f"Błąd podczas ładowania {file_path}: {e}")

        print(f"Załadowano {len(populations)} generacji z {len(files)} plików")
        return populations

    def extract_fitness_data(
        self, populations: List[Dict]
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """
        Ekstraktuje dane fitness z populacji

        Returns:
            generations: Lista numerów generacji
            best_fitness: Lista najlepszych fitness dla każdej generacji
            avg_fitness: Lista średnich fitness dla każdej generacji
            worst_fitness: Lista najgorszych fitness dla każdej generacji
        """
        generations = []
        best_fitness = []
        avg_fitness = []
        worst_fitness = []

        for pop_data in populations:
            gen_num = pop_data.get("nr_generacji", 0)
            osobnicy = pop_data.get("osobnicy", [])

            if not osobnicy:
                continue

            fitness_values = [ind.get("fitness", 0) for ind in osobnicy]

            generations.append(gen_num)
            best_fitness.append(max(fitness_values))
            avg_fitness.append(np.mean(fitness_values))
            worst_fitness.append(min(fitness_values))

        return generations, best_fitness, avg_fitness, worst_fitness

    def plot_fitness_progress(
        self,
        save_path: str = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        Generuje wykres postępu fitness

        Args:
            save_path: Ścieżka do zapisania wykresu (opcjonalnie)
            show_plot: Czy pokazać wykres
            figsize: Rozmiar wykresu (szerokość, wysokość)
        """
        populations = self.load_population_data()
        if not populations:
            return

        generations, best_fitness, avg_fitness, worst_fitness = (
            self.extract_fitness_data(populations)
        )

        if not generations:
            print("Brak danych do wykresu!")
            return

        # Tworzenie wykresu
        plt.figure(figsize=figsize)

        # Wykres główny
        plt.plot(
            generations,
            best_fitness,
            "g-",
            linewidth=2.5,
            label="Najlepszy fitness",
            marker="o",
            markersize=4,
        )
        plt.plot(
            generations,
            avg_fitness,
            "b--",
            linewidth=2,
            label="Średni fitness",
            marker="s",
            markersize=3,
        )
        plt.plot(
            generations,
            worst_fitness,
            "r:",
            linewidth=1.5,
            label="Najgorszy fitness",
            marker="^",
            markersize=3,
        )

        # Wypełnienie obszaru między najlepszym a najgorszym
        plt.fill_between(
            generations,
            best_fitness,
            worst_fitness,
            alpha=0.2,
            color="gray",
            label="Rozrzut populacji",
        )

        # Formatowanie wykresu
        plt.title("Postęp uczenia - Fitness w czasie", fontsize=16, fontweight="bold")
        plt.xlabel("Generacja", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Dodaj statystyki
        if generations:
            max_gen = max(generations)
            final_best = best_fitness[-1] if best_fitness else 0
            final_avg = avg_fitness[-1] if avg_fitness else 0
            improvement = (
                best_fitness[-1] - best_fitness[0] if len(best_fitness) > 1 else 0
            )

            stats_text = f"Generacja: {max_gen}\nNajlepszy: {final_best:.2f}\nŚredni: {final_avg:.2f}\nPoprawa: {improvement:.2f}"
            plt.text(
                0.02,
                0.98,
                stats_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        # Zapisz wykres jeśli podano ścieżkę
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Wykres zapisany jako: {save_path}")

        # Pokaż wykres
        if show_plot:
            plt.show()

    def plot_fitness_distribution(
        self, generation: int = None, save_path: str = None, show_plot: bool = True
    ) -> None:
        """
        Generuje histogram rozkładu fitness dla konkretnej generacji

        Args:
            generation: Numer generacji (None = ostatnia)
            save_path: Ścieżka do zapisania wykresu
            show_plot: Czy pokazać wykres
        """
        populations = self.load_population_data()
        if not populations:
            return

        # Wybierz generację
        if generation is None:
            target_gen = populations[-1]
            gen_num = target_gen.get("nr_generacji", len(populations) - 1)
        else:
            target_gen = None
            for pop in populations:
                if pop.get("nr_generacji") == generation:
                    target_gen = pop
                    gen_num = generation
                    break

            if target_gen is None:
                print(f"Nie znaleziono generacji {generation}")
                return

        osobnicy = target_gen.get("osobnicy", [])
        if not osobnicy:
            print(f"Brak danych dla generacji {gen_num}")
            return

        fitness_values = [ind.get("fitness", 0) for ind in osobnicy]

        # Tworzenie histogramu
        plt.figure(figsize=(10, 6))

        n_bins = min(20, len(set(fitness_values)))  # Maksymalnie 20 binów
        plt.hist(
            fitness_values, bins=n_bins, alpha=0.7, color="skyblue", edgecolor="black"
        )

        # Dodaj linie statystyk
        mean_fitness = np.mean(fitness_values)
        median_fitness = np.median(fitness_values)

        plt.axvline(
            mean_fitness,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Średnia: {mean_fitness:.2f}",
        )
        plt.axvline(
            median_fitness,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mediana: {median_fitness:.2f}",
        )

        plt.title(
            f"Rozkład fitness - Generacja {gen_num}", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Fitness", fontsize=12)
        plt.ylabel("Liczba osobników", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Dodaj statystyki
        stats_text = f"Populacja: {len(fitness_values)}\nMin: {min(fitness_values):.2f}\nMax: {max(fitness_values):.2f}\nStd: {np.std(fitness_values):.2f}"
        plt.text(
            0.98,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Histogram zapisany jako: {save_path}")

        if show_plot:
            plt.show()

    def get_training_summary(self) -> Dict:
        """
        Zwraca podsumowanie treningu

        Returns:
            Słownik z kluczowymi statystykami
        """
        populations = self.load_population_data()
        if not populations:
            return {}

        generations, best_fitness, avg_fitness, worst_fitness = (
            self.extract_fitness_data(populations)
        )

        if not generations:
            return {}

        summary = {
            "total_generations": len(generations),
            "last_generation": max(generations),
            "initial_best_fitness": best_fitness[0] if best_fitness else 0,
            "final_best_fitness": best_fitness[-1] if best_fitness else 0,
            "improvement": (
                best_fitness[-1] - best_fitness[0] if len(best_fitness) > 1 else 0
            ),
            "final_avg_fitness": avg_fitness[-1] if avg_fitness else 0,
            "max_fitness_achieved": max(best_fitness) if best_fitness else 0,
            "min_fitness_achieved": min(worst_fitness) if worst_fitness else 0,
            "convergence_rate": self._calculate_convergence_rate(best_fitness),
        }

        return summary

    def _calculate_convergence_rate(self, fitness_values: List[float]) -> float:
        """Oblicza tempo zbieżności (poprawa na generację)"""
        if len(fitness_values) < 2:
            return 0.0

        improvements = []
        for i in range(1, len(fitness_values)):
            improvement = fitness_values[i] - fitness_values[i - 1]
            improvements.append(max(0, improvement))  # Tylko pozytywne poprawy

        return np.mean(improvements) if improvements else 0.0

    def monitor_training(
        self, refresh_interval: int = 10, auto_save: bool = True
    ) -> None:
        """
        Tryb monitorowania treningu w czasie rzeczywistym

        Args:
            refresh_interval: Interwał odświeżania w sekundach
            auto_save: Czy automatycznie zapisywać wykresy
        """
        import time

        print("Rozpoczęto monitorowanie treningu...")
        print("Naciśnij Ctrl+C aby zakończyć")

        last_gen_count = 0

        try:
            while True:
                populations = self.load_population_data()
                current_gen_count = len(populations)

                if current_gen_count > last_gen_count:
                    print(f"\nNowa generacja wykryta! ({current_gen_count} generacji)")

                    # Generuj wykres
                    save_path = (
                        f"fitness_progress_gen_{current_gen_count-1:03d}.png"
                        if auto_save
                        else None
                    )
                    self.plot_fitness_progress(save_path=save_path, show_plot=False)

                    # Wyświetl podsumowanie
                    summary = self.get_training_summary()
                    if summary:
                        print(f"Najlepszy fitness: {summary['final_best_fitness']:.2f}")
                        print(f"Średni fitness: {summary['final_avg_fitness']:.2f}")
                        print(f"Poprawa: {summary['improvement']:.2f}")

                    last_gen_count = current_gen_count

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nMonitorowanie zakończone przez użytkownika")


def main():
    """Przykład użycia"""
    # Utwórz katalog populations jeśli nie istnieje
    os.makedirs("populations", exist_ok=True)

    plotter = FitnessPlotter("populations")

    print("=== TAXI GENETIC ALGORITHM - FITNESS PLOTTER ===\n")

    while True:
        print("Wybierz opcję:")
        print("1. Wygeneruj wykres postępu fitness")
        print("2. Wygeneruj histogram rozkładu fitness")
        print("3. Pokaż podsumowanie treningu")
        print("4. Tryb monitorowania w czasie rzeczywistym")
        print("5. Wyjście")

        choice = input("\nWybór (1-5): ").strip()

        if choice == "1":
            print("\nGenerowanie wykresu postępu fitness...")
            save = input("Zapisać wykres? (t/n): ").lower() == "t"
            save_path = "fitness_progress.png" if save else None
            plotter.plot_fitness_progress(save_path=save_path)

        elif choice == "2":
            gen_input = input("Podaj numer generacji (Enter = ostatnia): ").strip()
            generation = int(gen_input) if gen_input.isdigit() else None

            save = input("Zapisać histogram? (t/n): ").lower() == "t"
            save_path = (
                f"fitness_distribution_gen_{generation or 'last'}.png" if save else None
            )
            plotter.plot_fitness_distribution(
                generation=generation, save_path=save_path
            )

        elif choice == "3":
            print("\nPodsumowanie treningu:")
            summary = plotter.get_training_summary()
            if summary:
                print(f"Całkowita liczba generacji: {summary['total_generations']}")
                print(f"Ostatnia generacja: {summary['last_generation']}")
                print(
                    f"Początkowy najlepszy fitness: {summary['initial_best_fitness']:.2f}"
                )
                print(f"Końcowy najlepszy fitness: {summary['final_best_fitness']:.2f}")
                print(f"Całkowita poprawa: {summary['improvement']:.2f}")
                print(f"Końcowy średni fitness: {summary['final_avg_fitness']:.2f}")
                print(
                    f"Maksymalny osiągnięty fitness: {summary['max_fitness_achieved']:.2f}"
                )
                print(f"Minimalny fitness: {summary['min_fitness_achieved']:.2f}")
                print(f"Tempo zbieżności: {summary['convergence_rate']:.2f}/generacja")
            else:
                print("Brak danych do podsumowania")

        elif choice == "4":
            interval = input(
                "Interwał odświeżania w sekundach (domyślnie 10): "
            ).strip()
            interval = int(interval) if interval.isdigit() else 10

            auto_save = input("Automatycznie zapisywać wykresy? (t/n): ").lower() == "t"
            plotter.monitor_training(refresh_interval=interval, auto_save=auto_save)

        elif choice == "5":
            print("Do widzenia!")
            break

        else:
            print("Nieprawidłowy wybór!")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
