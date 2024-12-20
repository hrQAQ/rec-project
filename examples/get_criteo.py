import pandas as pd

SAMPLE_DATA_PATH = "/root/DeepCTR-Torch/examples/criteo_sample.txt"
TRAIN_DATA_PATH = "/root/DeepCTR-Torch/data/train.txt"

# 参数为 sample, part, total
def get_criteo_dataset(pattern, sample_num=10000):
    total_num = 45840617
    sample_num = int(total_num * 0.05)
    names = ['label'] + ['I' + str(i) for i in range(1, 14)] + ['C' + str(i) for i in range(1, 27)]
    if pattern == "sample":
        data = pd.read_csv(SAMPLE_DATA_PATH)
    if pattern == "part":
        print("sample_num: ", sample_num)
        data = pd.read_csv(TRAIN_DATA_PATH, sep='\t', iterator=True, header=None,
                           names=names)
        data = data.get_chunk(sample_num)
    if pattern == "total":
        data = pd.read_csv(TRAIN_DATA_PATH, sep='\t', header=None, names=names)
    return data

if __name__ == "__main__":
    data = get_criteo_dataset('part')
    print(data.head())
    print(data.shape)