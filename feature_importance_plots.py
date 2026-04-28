import optuna
import matplotlib.pyplot as plt
import os

direc = "./MedT_optuna_results"
db_path = f"sqlite:///{os.path.join(direc, 'MedT_HPO.db')}"
study_name = "MedT_HPO"

print(f"Loading study '{study_name}' from database...")
study = optuna.load_study(study_name=study_name, storage=db_path)

print("Calculating feature importance...")
optuna.visualization.matplotlib.plot_param_importances(study)

save_path = os.path.join(direc, "MedT_parameter_importance.png")
plt.tight_layout()
plt.savefig(save_path)
print(f"Feature importance graph saved to: {save_path}")
