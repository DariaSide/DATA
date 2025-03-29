import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import Descriptors
from deap import base, creator, tools, algorithms

# Загрузка данных
df = pd.read_csv('C:\\Users\\Alksq\\Downloads\\database_CCDC.csv', header=None, names=['SMILES'])
initial_smiles = df['SMILES'].tolist()

# Инициализация RDKit
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return mol

# Фитнес-функция (пример: максимизация молекулярной массы и количества водородных связей)
def evaluate(mol):
    try:
        h_bond_donors = Descriptors.NumHDonors(mol)
        molecular_weight = Descriptors.MolWt(mol)
        return (h_bond_donors + molecular_weight/1000),
    except:
        return (0,)

# Загрузка фрагментов из файла с указанием кодировки UTF-8
with open('C:\\Users\\Alksq\\OneDrive\\Рабочий стол\\functions.txt', 'r', encoding='utf-8') as f:
    fragments = [
        line.split('#')[0].strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

# Проверка валидности фрагментов
valid_fragments = []
for smi in fragments:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid_fragments.append(smi)
fragments = valid_fragments


def mutate_smiles(individual):
    smiles = individual
    try:
        mol = smiles_to_mol(smiles)
    except ValueError:
        # Возвращаем исходный индивид, если SMILES невалиден
        return (creator.Individual(smiles),)

    if not fragments:
        raise ValueError("No valid fragments loaded!")

    if random.random() < 0.5:
        # Добавление фрагмента с проверкой валидности
        for _ in range(10):  # Попытки найти подходящий фрагмент
            fragment = random.choice(fragments)
            mol_fragment = Chem.MolFromSmiles(fragment)
            if mol_fragment is None:
                continue
            try:
                new_mol = Chem.CombineMols(mol, mol_fragment)
                new_smiles = Chem.MolToSmiles(new_mol)
                if Chem.MolFromSmiles(new_smiles) is not None:
                    break
            except:
                continue
        else:
            new_smiles = smiles  # Если не удалось найти валидный вариант
    else:
        # Удаление атома с проверкой валидности
        new_smiles = smiles
        if mol.GetNumAtoms() > 1:
            for _ in range(10):  # Попытки удалить атом без ошибок
                editable_mol = Chem.EditableMol(mol)
                atoms = list(editable_mol.GetMol().GetAtoms())
                if not atoms:
                    break
                atom_to_remove = random.choice(atoms)
                editable_mol.RemoveAtom(atom_to_remove.GetIdx())
                new_mol = editable_mol.GetMol()
                new_smiles_tmp = Chem.MolToSmiles(new_mol)
                if Chem.MolFromSmiles(new_smiles_tmp) is not None:
                    new_smiles = new_smiles_tmp
                    break

    return (creator.Individual(new_smiles),)


def cx_smiles(ind1, ind2):
    # Проверка валидности родителей
    try:
        mol1 = smiles_to_mol(ind1)
        mol2 = smiles_to_mol(ind2)
    except ValueError:
        return (ind1, ind2)  # Возвращаем исходных индивидов при ошибке

    # Попытки создать валидных потомков
    for _ in range(10):
        combo1 = Chem.CombineMols(mol1, mol2)
        combo2 = Chem.CombineMols(mol2, mol1)
        smiles1 = Chem.MolToSmiles(combo1)
        smiles2 = Chem.MolToSmiles(combo2)
        if Chem.MolFromSmiles(smiles1) and Chem.MolFromSmiles(smiles2):
            return (
                creator.Individual(smiles1),
                creator.Individual(smiles2)
            )

    # Если не удалось, возвращаем исходных индивидов
    return (ind1, ind2)
# Настройка DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", str, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", creator.Individual, random.choice(initial_smiles))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cx_smiles)  # Теперь возвращает двух потомков
toolbox.register("mutate", mutate_smiles)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", lambda x: evaluate(smiles_to_mol(x)))

# Параметры эволюции
population = toolbox.population(n=50)
NGEN = 20
result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=NGEN, verbose=False)

# Сбор результатов
best_smiles = [ind for ind in population if not ind.fitness.values == (0,)]
output_df = pd.DataFrame({'SMILES': best_smiles})
output_filename = 'optimized_theophylline_coformers.csv'
output_df.to_csv(output_filename, index=False)

print(f"Результаты сохранены в {output_filename}")
print(f"Сгенерировано улучшенных молекул: {len(output_df)}")