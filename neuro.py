import matplotlib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, SanitizeFlags
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity
import random
import warnings
from rdkit import RDLogger
from tqdm import tqdm
from functools import lru_cache
import multiprocessing as mp
from rdkit.Chem import BRICS
import matplotlib.pyplot as plt
import dill
import logging
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

logging.basicConfig(filename='ga_errors.log', level=logging.INFO)
logging.basicConfig(filename='ga_debug.log', level=logging.DEBUG)

# Параметры алгоритма
POPULATION_SIZE = 800
GENERATIONS = 150
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.5
ELITE_SIZE = 50
TOURNAMENT_SIZE = 30
N_CORES = mp.cpu_count()   # Используем половину ядер

# Оптимизированные параметры фингерпринтов
FP_RADIUS = 2
FP_BITS = 2048  # Уменьшен размер фингерпринта

# Кэш для молекул и фингерпринтов
@lru_cache(maxsize=10_000)
def get_cached_mol(smiles):
    return Chem.MolFromSmiles(smiles)

@lru_cache(maxsize=10_000)
def get_cached_fingerprint(smiles):
    mol = get_cached_mol(smiles)
    return GetMorganFingerprintAsBitVect(mol, FP_RADIUS, FP_BITS) if mol else None

# Отключение предупреждений
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

# Параллельные вычисления
def parallel_calculate(args):
    try:
        smiles, target_fp = args
        mol = get_cached_mol(smiles)
        if not mol:
            return 0.0
        fp = get_cached_fingerprint(smiles)
        similarity = TanimotoSimilarity(fp, target_fp)
        logP = Descriptors.MolLogP(mol)
        return 0.6 * similarity + 0.4 * (1 - abs(logP - 1.5)/2)
    except Exception as e:
        print(f"Ошибка в parallel_calculate: {e}")
        return 0.0

def calculate_fitness_parallel(population, target_fp):
    ctx = mp.get_context('spawn')
    ctx.reducer = dill.Reduce  # Используем dill для сериализации
    with ctx.Pool(N_CORES) as pool:
        args = [(smiles, target_fp) for smiles in population]
        return list(pool.imap(parallel_calculate, args))

def sanitize_mol(mol):
    """Улучшенная санитизация молекулы"""
    try:
        flags = SanitizeFlags.SANITIZE_ALL
        Chem.SanitizeMol(mol, flags)
        return mol
    except:
        return None


def validate_molecule(smiles):
    """Полная проверка валидности молекулы"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    try:
        for atom in mol.GetAtoms():
            if atom.GetExplicitValence() > Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()):
                return False
        return True
    except:
        return False

def combine_with_scaffold(mol, scaffold_frag):
    """Комбинирует исходную молекулу с фрагментом целевого каркаса"""
    try:
        combined = Chem.CombineMols(mol, scaffold_frag)
        ed_combined = Chem.EditableMol(combined)
        # Соединяем случайные атомы из исходной молекулы и фрагмента
        anchor1 = random.randint(0, mol.GetNumAtoms()-1)
        anchor2 = random.randint(mol.GetNumAtoms(), combined.GetNumAtoms()-1)
        ed_combined.AddBond(anchor1, anchor2, Chem.BondType.SINGLE)
        new_mol = ed_combined.GetMol()
        new_mol = sanitize_mol(new_mol)
        return new_mol
    except:
        return mol





def crossover(smiles1, smiles2):
    try:
        mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
        if not mol1 or not mol2:
            return (smiles1, smiles2)

        # Глубокий поиск фрагментов
        frags1 = list(BRICS.BRICSDecompose(mol1, minFragmentSize=4))  # Минимум 4 атома
        frags2 = list(BRICS.BRICSDecompose(mol2, minFragmentSize=4))

        if not frags1 or not frags2:
            return (smiles1, smiles2)

        frag1 = random.choice(frags1)
        frag2 = random.choice(frags2)
        combined = BRICS.CombineFragments(frag1, frag2)
        return (Chem.MolToSmiles(combined),)
    except:
        return (smiles1, smiles2)


def rank_selection(population, fitness):
    ranked = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked[:int(0.2*POPULATION_SIZE)]]



def tournament_selection(population, fitness):
    candidates = random.sample(list(zip(population, fitness)), 5)
    return max(candidates, key=lambda x: x[1])[0]


class MolecularOptimizer:
    def __init__(self, target_smiles, population_size=2000):
        # Инициализация молекулы один раз
        self.target_mol = Chem.MolFromSmiles(target_smiles)
        if not self.target_mol:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        # Инициализация остальных параметров
        self.target_fp = GetMorganFingerprintAsBitVect(self.target_mol, FP_RADIUS, FP_BITS)
        self.target_frags = list(BRICS.BRICSDecompose(self.target_mol))  # Убрано дублирование
        self.population_size = population_size
        self.population = []
        self.fitness_history = []
        self.params = {
            'generations': GENERATIONS,
            'mutation_rate': MUTATION_RATE,
            'crossover_rate': CROSSOVER_RATE
        }

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'target_fp': self.target_fp,
                'population': self.population,
                'fitness_history': self.fitness_history,
                'params': self.params
            }, f)

    @classmethod
    def load(cls, filename):
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.__dict__.update(data)
        return instance

    def calculate_fitness(self, smiles):
        if not validate_molecule(smiles):
            return 0.0

        mol = Chem.MolFromSmiles(smiles)
        fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        similarity = TanimotoSimilarity(fp, self.target_fp)

        # Критерии для коформов:
        # - Наличие доноров/акцепторов водородных связей
        # - Возможность образования π-π взаимодействий
        # - Оптимальная стерическая упаковка

        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)

        # Бонусы за взаимодействия
        h_bond_bonus = 0.1 * min(h_donors + h_acceptors, 4)  # До 4 взаимодействий
        pi_pi_bonus = 0.15 * aromatic_rings if aromatic_rings >= 2 else 0

        # Штрафы за несовместимые параметры
        mw_penalty = 0.02 * (Descriptors.MolWt(mol) / 500 if Descriptors.MolWt(mol) > 500 else 0)

        return 0.7 * similarity + 0.2 * (h_bond_bonus + pi_pi_bonus) - mw_penalty

    def optimize(self, initial_data_path):
        self.population = random.sample(smiles_list, self.population_size)

        add_counter = 0
        add_interval = random.randint(2, 3)
        best_fitness_history = []
        current_mutation_rate = self.params['mutation_rate']  # Используем параметры класса

        # Один цикл по количеству поколений
        for generation in tqdm(range(self.params['generations']), desc="Evolution"):
            # Фильтрация и оценка
            self.population = [s for s in self.population if validate_molecule(s)]
            fitness = [self.calculate_fitness(s) for s in self.population]
            current_best = max(fitness) if fitness else 0.0
            self.fitness_history.append(current_best)

            # Проверка застоя
            if len(self.fitness_history) > 5 and (self.fitness_history[-1] - self.fitness_history[-5]) < 0.01:
                current_mutation_rate = 0.8  # Резкое увеличение при застое
            else:
                current_mutation_rate = self.params['mutation_rate']

            # Пример добавления новых молекул при низком разнообразии
            if len(self.population) < 0.5 * self.population_size:
                new_samples = random.sample(smiles_list, self.population_size // 3)
                self.population = list(set(self.population + new_samples))[:self.population_size]

            # Адаптивная мутация
            if current_best < 0.5:
                current_mutation_rate = min(0.6, current_mutation_rate + 0.1)
            else:
                current_mutation_rate = max(0.2, current_mutation_rate - 0.05)

            # Элитарный отбор
            elite_indices = np.argsort(fitness)[-ELITE_SIZE:]
            elites = [self.population[i] for i in elite_indices]

            # Генерация потомков
            offspring = []
            while len(offspring) < self.population_size - ELITE_SIZE:
                parent1 = tournament_selection(self.population, fitness)
                parent2 = tournament_selection(self.population, fitness)
                child1, child2 = crossover(parent1, parent2)
                offspring.extend([child1, child2])

            # Мутация элит
            mutated_elites = [self.mutate(s) if random.random() < 0.8 else s for s in elites]
            offspring = mutated_elites + offspring

            # Обновление популяции
            self.population = list({s for s in offspring if validate_molecule(s)})[:self.population_size]

            # Добавление новых образцов
            add_counter += 1
            # В методе optimize:
            # В методе optimize, блок добавления целевых фрагментов:
            if generation % 3 == 0:
                target_frags = []
                for frag in self.target_frags:
                    try:
                        # Проверяем, что frag - объект Mol, и конвертируем в SMILES
                        if isinstance(frag, Chem.rdchem.Mol):
                            smiles = Chem.MolToSmiles(frag)
                            if validate_molecule(smiles):
                                target_frags.append(smiles)
                        else:
                            logging.warning("Фрагмент не является объектом Mol")
                    except Exception as e:
                        logging.error(f"Ошибка преобразования фрагмента: {e}")
                        continue

                if target_frags:
                    # Весовой отбор фрагментов, похожих на текущего лидера
                    leader_smiles = max(zip(self.population, fitness), key=lambda x: x[1])[0]
                    leader_fp = get_cached_fingerprint(leader_smiles)
                    weights = [TanimotoSimilarity(get_cached_fingerprint(s), leader_fp) for s in target_frags]
                    new_samples = random.choices(target_frags, weights=weights, k=int(0.1 * self.population_size))
                    self.population = list(set(self.population + new_samples))[:self.population_size]

            print(f"Gen {generation}: Best {current_best:.2f} Diversity {len(self.population)}")
            # Логирование топ-5 молекул (добавить эти строки)
            top_molecules = sorted(zip(self.population, fitness), key=lambda x: x[1], reverse=True)[:5]
            print("Топ-5 молекул:")
            for idx, (sm, fit) in enumerate(top_molecules):
                print(f"  {idx + 1}. {sm} | Fitness: {fit:.2f}")

        # Сохранение графика
        if len(self.fitness_history) > 0:
            plt.plot(self.fitness_history)
            plt.savefig('fitness_progress.png')
        return self.population

        top_mols = [Chem.MolFromSmiles(sm) for sm, _ in top_molecules[:5]]
        img = Draw.MolsToGridImage(top_mols, molsPerRow=5, subImgSize=(300, 300))
        img.save(f'gen_{generation}_top5.png')

    def mutate(self, smiles):  # self добавлен как параметр
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return smiles

        new_mol = Chem.RWMol(mol)
        try:
            mutation_type = random.choice([
                "replace_atom", "add_bond", "remove_atom",
                "add_ring", "add_functional_group", "scaffold_hopping",
                "change_bond_type"])
            mutation_type = random.choices(
                ["scaffold_hopping", "replace_atom", "add_functional_group", "add_ring"],
                weights=[0.5, 0.2, 0.2, 0.1],  # Приоритет scaffold_hopping
                k=1
            )[0]

            if mutation_type == "replace_atom":
                atom_idx = random.choice(range(new_mol.GetNumAtoms()))
                atom = new_mol.GetAtomWithIdx(atom_idx)
                new_element = random.choice(['C', 'N', 'O'])
                new_atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(new_element)

            elif mutation_type == "add_bond":
                # Добавление случайной связи
                atoms = [atom.GetIdx() for atom in new_mol.GetAtoms()]
                if len(atoms) >= 2:
                    pair = random.sample(atoms, 2)
                    new_mol.AddBond(pair[0], pair[1], Chem.BondType.SINGLE)
            elif mutation_type == "remove_atom":
                # Удаление случайного атома (если возможно)
                if new_mol.GetNumAtoms() > 1:
                    atom_idx = random.choice(range(new_mol.GetNumAtoms()))
                    new_mol.RemoveAtom(atom_idx)

            elif mutation_type == "add_ring":
                # 1. Выбрать случайный атом для присоединения кольца
                atoms = [atom for atom in new_mol.GetAtoms()]
                if not atoms:
                    return smiles
                anchor_atom = random.choice(atoms)
                anchor_idx = anchor_atom.GetIdx()

                # 2. Создать бензольное кольцо
                ring = Chem.MolFromSmiles("C1=CC=CC=C1")  # Бензол
                if not ring:
                    return smiles

                # 3. Присоединить кольцо к выбранному атому
                combined = Chem.CombineMols(new_mol, ring)
                ed_combined = Chem.EditableMol(combined)
                ed_combined.AddBond(anchor_idx, len(new_mol.GetAtoms()),
                                    Chem.BondType.SINGLE)  # Связь между атомом и кольцом

                # 4. Санитизация и проверка валидности
                modified_mol = ed_combined.GetMol()
                modified_mol = sanitize_mol(modified_mol)
                if modified_mol and validate_molecule(Chem.MolToSmiles(modified_mol)):
                    return Chem.MolToSmiles(modified_mol)  # Возврат здесь


            elif mutation_type == "add_functional_group":
                if new_mol.GetNumAtoms() == 0:
                    return smiles
                atom_idx = random.choice(range(new_mol.GetNumAtoms()))
                group = random.choice(["O", "N", "F", "Cl"])  # Простые атомы вместо SMARTS
                new_atom = Chem.Atom(group)
                new_mol.AddAtom(new_atom)
                new_mol.AddBond(atom_idx, new_mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)



            elif mutation_type == "scaffold_hopping":
                if new_mol.GetNumAtoms() > 5 and hasattr(self, 'target_frags') and len(self.target_frags) > 0:
                    scaffold_frag = random.choice(self.target_frags)
                    modified_mol = combine_with_scaffold(new_mol, scaffold_frag)
                    if modified_mol and validate_molecule(Chem.MolToSmiles(modified_mol)):
                        return Chem.MolToSmiles(modified_mol)

            new_smiles = Chem.MolToSmiles(sanitize_mol(new_mol))
            return new_smiles if validate_molecule(new_smiles) else smiles
        except Exception as e:
            logging.debug(f"Mutation failed for {smiles}: {str(e)}")
            print(f"Ошибка в mutate: {e}")
            return smiles



# Инициализация данных
data = pd.read_csv("C:\\Users\\Alksq\\Downloads\\database_CCDC.csv", header=None, names=['smiles'])
smiles_list = data['smiles'].tolist()
# Загрузка и продолжение работы


if __name__ == "__main__":
    mp.freeze_support()

    # Загрузка данных
    data = pd.read_csv("C:\\Users\\Alksq\\Downloads\\database_CCDC.csv", header=None, names=['smiles'])
    smiles_list = data['smiles'].tolist()

    # Инициализация и запуск
    optimizer = MolecularOptimizer(target_smiles='CN1C(=O)N(C)C2=C1N=CN2C')
    results = optimizer.optimize("C:\\Users\\Alksq\\Downloads\\database_CCDC.csv")

    # Сохранение
    optimizer.save('theophylline_optimizer.pkl')
    pd.DataFrame(results, columns=['smiles']).to_csv('results.csv', index=False)



