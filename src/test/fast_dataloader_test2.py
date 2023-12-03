# NOTE
#     : https://dev.classmethod.jp/articles/vscode_python_import_error/
#       pythonの実行時の探索パスと、VSCodeの探索パスは違うので、別途追加しないといけない。
import fast_dataloader as fdl
import _cshogi as cs
import numpy as np
import torch

np.set_printoptions(threshold = 100 * cs.N_FEATURE_WHC)
batch_size = 4

dataloader = fdl.FastDataloader("test/tmp_data2", batch_size)

print("info: np.empty()")
torch_features = torch.empty(
    (batch_size, cs.N_FEATURE_CHANNEL, cs.N_FEATURE_HEIGHT, cs.N_FEATURE_WIDTH),
    dtype=torch.float32, pin_memory=True
)
torch_probabilitys = torch.empty((batch_size, cs.N_LABEL_SIZE), dtype=torch.float32, pin_memory=True)
torch_values = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
torch_results = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

# データは、numpy配列に入れる。
# 配列に配列を代入しなければ(配列の要素に代入する限り)、メモリ領域はtorch.Tensor配列と共有されたまま。
features = torch_features.numpy()
probabilitys = torch_probabilitys.numpy()
values = torch_values.numpy()
results = torch_results.numpy()


print("info: start readin all file")
read_status = dataloader.read_files_all()
if not read_status:
    print("Error: failed to read files")
    exit(1)
idxs_list = [i for i in range(dataloader.size_hcpex())]

print(f"info: idxs_list = {idxs_list}")

dataloader.store_hcpex_idxs(
    features,
    probabilitys,
    values,
    results,
    idxs_list
)

print(f"info: values = {values}")
print(f"info: results = {results}")

for torch_probability in torch_probabilitys:
    print("==============================")
    for i in range(cs.N_LABEL_SIZE):
        p = torch_probability[i].item()
        if p > 0:
            print(f"    [{i}] = {p}")