import sys
import importlib.util
import os
import matplotlib.pyplot as plt

module_path = os.path.abspath("AI laby/lista 3/run03-finetune-example.py")
spec = importlib.util.spec_from_file_location("run03_finetune_example", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules["run03_finetune_example"] = module
spec.loader.exec_module(module)

lr_list = []
loss_list = []
for i in range(10):
    lr = 1e-4 * (2 ** i)
    print(f"Trenowanie z lr = {lr}")
    loss = module.train_model(lr)
    lr_list.append(lr)
    loss_list.append(loss)
    print(lr, loss)


plt.figure(figsize=(8, 5))
plt.plot(lr_list, loss_list, marker='o')
plt.xscale('log')
plt.xlabel('Learning rate')
plt.ylabel('Loss')
plt.title('Porównanie learning rate i loss')
plt.grid()
plt.show()