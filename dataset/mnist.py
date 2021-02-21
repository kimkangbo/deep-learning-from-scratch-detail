# coding: utf-8
# 참조: https://daddynkidsmakers.blogspot.com/2017/06/hello-world-mnist.html
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:   # wiht as 구문을 사용하면 파일을 열고 해당 구문이 끝나면 자동으로 닫히게 됩니다.
            labels = np.frombuffer(f.read(), np.uint8, offset=8) # 파일 처음기준으로 8byte 부터 0~9까지 값을 갖는 1byte 값을 배열로 저장함
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)  # 파일 처음기준으로 16byte 부터 pixel 0~255 값을 갖는 1byte 데이터를 읽음
    data = data.reshape(-1, img_size) # 1byte 1개의 배열로 된 것을 784byte list, 즉 784개의 list로 재구성함 [[0,...,783], [0,...,783], ...]

    print("Done")
    
    return data
    
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:	# https://wayhome25.github.io/cs/2017/04/04/cs-04/
        pickle.dump(dataset, f, -1)		# dump는 f에 데이터 입력하기
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10)) # https://icodebroker.tistory.com/5262 , X list 의 item 개수 만큼 10개의 zero 를 갖는 리스트를 생성함
    for idx, row in enumerate(T): # [[0,...0],[0,...,0],[],...] 에서 idx는 [0,...,0]의 index, row는 [0,...,0] 자체
        row[X[idx]] = 1           # row[특정인덱스] = 1로 지정하는데 특정인덱스 X[idx] = 0~9 값 중에 한개 즉, X[idx]=1일 경우 [0,1,0,...0]이 됨
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(save_file):	# 기존 pickle data file이 없는 경우
        init_mnist()					# 새로 pickle data file을 생성함
        
    with open(save_file, 'rb') as f:	# https://wayhome25.github.io/cs/2017/04/04/cs-04/
        dataset = pickle.load(f)		# load는 f에서 데이터 읽어 오기
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)	# https://rfriend.tistory.com/285
            dataset[key] /= 255.0							# list의 모든 값들을 255.0으로 나누어서 값을 0~1사이의 소수점으로 변환함
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28) # 784개의 1차원 배열들의 리스트를 28x28개의 2차원 배열들의 리스트로 변경함 [0,...,784] -> [[0,...,27],[0,...,27]]
#            dataset[key] = dataset[key].reshape(-1, 28, 28) # 784개의 1차원 배열들의 리스트를 28x28개의 2차원 배열들의 리스트로 변경함. 위의 것과 동일한 결과를 나타냄. 
			
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()
