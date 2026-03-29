import pickle

with open(r"C:\Users\BKAI\ducluong\DrugOptimization\CONAN-SchNet copy 2\data\processed\esol\seed_4\10_conformers\train.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys())
print(type(data['atomic_numbers']))
print(type(data['positions']))
print(len(data['atomic_numbers']))
print(len(data['positions']))
print(len(data['atomic_numbers'][0]))
print(len(data['positions'][0]))