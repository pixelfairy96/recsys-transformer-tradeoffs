from data.dataset import SequentialDataset
import yaml

with open("configs/sasrec_ml1m.yaml") as f:
    config = yaml.safe_load(f)

dataset = SequentialDataset(config)

print("Num items:", dataset.num_items)
print("Train examples:", len(dataset.get_train_data()))
print("Validation examples:", len(dataset.get_validation_data()))
print("Test examples:", len(dataset.get_test_data()))

print("Sample train example:", dataset.get_train_data()[0])