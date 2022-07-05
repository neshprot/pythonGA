import random
import matplotlib.pyplot as plt

# константы задача
ONE_MAX_LENGTH = 100   # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 200   # колличество индивидуумов в популяции
P_CROSSOVER = 0.9   # вероятность скрещивания
P_MUTATION = 0.1    # вероятность мутации индивидуума
MAX_GENERATIONS = 50    # максимальное количество поколений

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class FitnessMax():
    """приспособленность особи"""
    def __init__(self):
        self.values = [0]


class Individual(list):
    """индивиды в виде хромосом"""
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


def oneMaxFitness(individual):
    """вычисление приспособленности одной особи"""
    return sum(individual),  # кортеж


def individualCreator():
    """создание отдельного индивидуума"""
    return Individual([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])


def populationCreator(n=0):
    """создание популяции"""
    return list([individualCreator() for i in range(n)])


# вызов функции создания популяции
population = populationCreator(n=POPULATION_SIZE)
generationCounter = 0   # счётчик числа поколений

# вычисление и присваивание приспособленности каждой особи
fitnessValues = list(map(oneMaxFitness, population))    # список приспособленностей особей в текущей популяции

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

# списки для статистики
maxFitnessValues = []   # макс приспособленности
meanFitnessValues = []   # средняя приспособленность


# функция клонирования(т.к. в процессе отбора может быть 2 ссылки на одного индивидуума)
def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind


# турнирный отбор
def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring


# одноточечный кроссинговер
def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1)-3)    # точка разреза хромосомы(без попадания границ)
    child1[s:], child2[s:] = child2[s:], child1[s:]


# мутация indpb - вероятность мутации
def mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1    # инверсия бита


# коллекция из значений приспособленноси особей в данной популяции
fitnessValues = [individual.fitness.values[0] for individual in population]


# главный цикл работы GA
while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]): # 1-ый идет по четным, 2-ой по нечетным индексам
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant, indpb=1.0/ONE_MAX_LENGTH)

    freshFitnessValues = list(map(oneMaxFitness, offspring))    #обновление значений приспособленности особей новой популяции
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness.values[0] for ind in population]

    #формируем статистику
    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ. = {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")


#вывод статистики в виде графика
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()
