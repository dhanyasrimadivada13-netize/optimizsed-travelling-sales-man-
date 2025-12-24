# optimizsed-travelling-sales-man-
#Implemented Simulated Annealing to solve the Traveling Salesman Problem efficiently.
import random
import math
import matplotlib.pyplot as plt


class TSPSolver:
    """
    Traveling Salesman Problem solver using Simulated Annealing
    """

    def __init__(self, cities):
        """
        cities: List of (x, y) coordinates
        """
        self.cities = cities
        self.n = len(cities)

    def distance(self, i, j):
        """Euclidean distance between city i and j"""
        x1, y1 = self.cities[i]
        x2, y2 = self.cities[j]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def tour_length(self, tour):
        """Total distance of a tour"""
        total = 0
        for i in range(self.n):
            total += self.distance(tour[i], tour[(i + 1) % self.n])
        return total

    def initial_solution(self):
        """Nearest Neighbor heuristic"""
        unvisited = list(range(self.n))
        tour = [unvisited.pop(random.randint(0, self.n - 1))]

        while unvisited:
            last = tour[-1]
            next_city = min(unvisited, key=lambda c: self.distance(last, c))
            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def two_opt_swap(self, tour):
        """Reverse a random segment (2-opt move)"""
        i, j = sorted(random.sample(range(self.n), 2))
        new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
        return new_tour

    def simulated_annealing(self, temp=10000, cooling=0.995, iterations=50000):
        """
        Main optimization loop
        """
        current = self.initial_solution()
        current_cost = self.tour_length(current)

        best = current[:]
        best_cost = current_cost

        history = []

        for step in range(iterations):
            candidate = self.two_opt_swap(current)
            candidate_cost = self.tour_length(candidate)

            delta = candidate_cost - current_cost

            # Accept better solutions or probabilistically worse ones
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

            temp *= cooling
            history.append(best_cost)

        return best, best_cost, history

    def plot(self, tour, title="TSP Solution"):
        """Plot cities and final tour"""
        x = [self.cities[i][0] for i in tour] + [self.cities[tour[0]][0]]
        y = [self.cities[i][1] for i in tour] + [self.cities[tour[0]][1]]

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'bo-')
        plt.title(f"{title}\nTotal Distance: {self.tour_length(tour):.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


# ---------- Demo ----------
if __name__ == "__main__":
    random.seed(42)

    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(20)]

    solver = TSPSolver(cities)
    tour, dist, history = solver.simulated_annealing()

    print("Best Distance:", round(dist, 2))
    print("Best Tour:", tour)

    solver.plot(tour)

