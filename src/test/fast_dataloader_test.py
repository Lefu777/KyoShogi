# NOTE
#     : https://dev.classmethod.jp/articles/vscode_python_import_error/
#       pythonの実行時の探索パスと、VSCodeの探索パスは違うので、別途追加しないといけない。
import fast_dataloader as fdl
import _cshogi as cs
import numpy as np
import torch

np.set_printoptions(threshold = 100 * cs.N_FEATURE_WHC)
batch_size = 10

dataloader = fdl.FastDataloader("test/tmp_data", batch_size)
# dataloader = fdl.FastDataloader("tmp_data")
print("info: read_files")
rf = dataloader.read_files()
if(not rf) :
    exit(1)

print("info: print_teachers")
# dataloader.print_teachers()

print("info: np.empty()")
# NOTE
#     : dtype = np.float32 と指定しないといけなさそう。
#       dtype = np.float とかすると、精度がc++ におけるfloat より高くて、キャストするとおかしくなるのか、
#       出力の様子がおかしい。
features = np.empty(cs.N_FEATURE_WHC * batch_size, dtype = np.float32)
moves = np.empty((batch_size, 1), dtype = np.int32)
values = np.empty((batch_size, 1), dtype = np.float32)
results = np.empty((batch_size, 1), dtype = np.float32)

print("info: store_teachers()")
dataloader.store_teachers(
    features,
    moves,
    values,
    results,
    259
)

print("info: reshape()")
features = features.reshape(batch_size, cs.N_FEATURE_CHANNEL, 9, 9)

print("info: print()")
print(f"info: {features.shape}")
print(f"info: {features[0].shape}")
print(features[9])

moves_tmp = [cs.move_to_usi(move) for move in moves]
print(moves_tmp)
print(values)
print(results)