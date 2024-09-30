import os 
import pickle
from cfg import Config
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import time
from cfg import Config
from python_speech_features import mfcc

config = Config()


class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
                        
        
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                  out_channels=16,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=1) 
        self.conv_2 = torch.nn.Conv2d(in_channels=16,
                              out_channels=32,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=1) 
        self.conv_3 = torch.nn.Conv2d(in_channels=32,
                              out_channels=64,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=1) 
        self.conv_4 = torch.nn.Conv2d(in_channels=64,
                              out_channels=128,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=1)
        
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        

        self.linear_1 = torch.nn.Linear(3072, 128)

        self.linear_2 = torch.nn.Linear(128, 64)

        self.linear_3 = torch.nn.Linear(64, 10)

        
    def embedding(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.conv_2(out)
        out = F.relu(out)
        out = self.conv_3(out)
        out = F.relu(out)
        out = self.conv_4(out)
        out = F.relu(out)
        out = self.max_pool(out)
        out = out.flatten(start_dim=1, end_dim=-1)
        out = self.linear_1(out)
        out = F.relu(out)
        out = self.linear_2(out)
        embeddings = F.relu(out)
        
        return embeddings
        
        
    def forward(self, x):

        out = self.embedding(x)
        out = self.linear_3(out)
        logits = F.relu(out)
        
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
# ---------------------------------------------------------------------------------------------------------------------

def check_data():
    if os.path.isfile(config.p_path):
        print ('Loading existing data from {} model'.format(config.mode))
        with open(config.p_path,'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
    
    
def return_sample(dummy) :
    # Todo: currently hardcoded 
    probs = np.array([
        0.12549036, 0.0233644, 0.1103633, 0.13340387,
        0.06538173,0.15995033, 0.06469909, 0.15596201,
        0.0640102, 0.09737471
    ])
    rand_class = np.random.choice(class_dist.index,p=probs)
    file = np.random.choice(df[df.label==rand_class].index)
    rate, wav = wavfile.read('clean/'+file)
    wavfile.write("instruments/"+file, rate=rate, data=wav)
    label = df.at[file,'label']
    rand_index = np.random.randint(0,wav.shape[0]-config.step)
    sample = wav[rand_index:rand_index+config.step]
    X_sample = mfcc(sample,rate,numcep=config.nfeat
                   ,nfilt=config.nfilt,nfft=config.nfft)


    return X_sample,classes.index(label)


def return_sample_resample(dummy) :
    # Todo: currently hardcoded 
    probs = np.array([
        0.12549036, 0.0233644, 0.1103633, 0.13340387,
        0.06538173,0.15995033, 0.06469909, 0.15596201,
        0.0640102, 0.09737471
    ])
    rand_class = np.random.choice(class_dist.index,p=probs)
    file = np.random.choice(df[df.label==rand_class].index)
    mysterious_number = 4
    rate, wav = wavfile.read('clean/'+file)
    rate = int(rate/mysterious_number)
    wav = wav[::mysterious_number]
    wavfile.write("instruments/lebanon_"+file, rate=rate, data=wav)
    label = df.at[file,'label']
    step = int(config.step/mysterious_number)
    rand_index = np.random.randint(0,wav.shape[0]-step)
    sample = wav[rand_index:rand_index+step]
    X_sample = mfcc(sample,rate,numcep=config.nfeat
                   ,nfilt=config.nfilt,nfft=config.nfft)


    return X_sample,classes.index(label)

    
def build_rand_feat():
    tmp = check_data()
    if tmp:
        print ('data exist')
        return tmp.data[0], tmp.data[1]
    
    X = []
    y = []
    _min,_max = float('inf'),-float('inf')
    import multiprocessing
    pool = multiprocessing.Pool(4)
    total_samples = 200000
    results= list(tqdm(pool.imap_unordered(return_sample,range(total_samples)),total=total_samples))
    pool.close()

    for result in results :
        _min = min(np.amin(result[0]),_min)
        _max = max(np.amin(result[0]),_max)
        X.append(result[0])
        y.append(result[1])


    config.min = _min
    config.max = _max
    X,y = np.array(X), np.array(y)
    X = (X - _min) / (_max-_min)

    config.data = (X,y)
    
    with open('./pickels/conv.p','wb') as handle:
        pickle.dump(config,handle,protocol=2)
        
    return X,y


def mysterious_lebanese_method():
    X = []
    y = []
    _min,_max = float('inf'),-float('inf')
    import multiprocessing
    pool = multiprocessing.Pool(4)
    total_samples = 200000
    results= list(tqdm(pool.imap_unordered(return_sample_resample,range(total_samples)),total=total_samples))
    pool.close()

    
    for result in results :
        _min = min(np.amin(result[0]),_min)
        _max = max(np.amin(result[0]),_max)
        X.append(result[0])
        y.append(result[1])


    config.min = _min
    config.max = _max
    X,y = np.array(X), np.array(y)
    X = (X - _min) / (_max-_min)
        
    return X,y

@torch.no_grad()
def compute_scores(model, data_loader):
    correct_pred, num_examples = 0, 0
    probas = torch.empty(0)
    predicted_labels = torch.empty(0)
    targets = torch.empty(0)
    for features, cur_targets in data_loader:
        features = features.to(config.device).float()
        cur_targets = cur_targets.to(config.device)
        cur_logits, cur_probas = model(features)
        _, cur_predicted_labels = torch.max(cur_probas, 1)
        probas = torch.cat((probas, cur_probas), dim = 0)
        predicted_labels = torch.cat((predicted_labels, cur_predicted_labels), dim = 0)
        targets = torch.cat((targets, cur_targets), dim = 0)
    return probas, predicted_labels, targets

@torch.no_grad()
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.to(config.device).float()
        targets = targets.to(config.device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

@torch.no_grad()
def compute_embeddings(model, data_loader, num_batches=None):
    embeddings = torch.empty(0)
    targets = torch.empty(0)
    for i, (features, cur_targets) in enumerate(data_loader):
        if num_batches is not None and i>=num_batches:
            continue
        features = features.to(config.device).float()
        cur_targets = cur_targets.to(config.device)
        cur_embeddings = model.embedding(features)
        embeddings = torch.cat((embeddings, cur_embeddings), dim = 0)
        targets = torch.cat((targets, cur_targets), dim = 0)
    return embeddings, targets

@torch.no_grad()
def compute_batch_embeddings(model, features, targets):
    features = features.to(config.device).float()
    targets = targets.to(config.device)
    embeddings = model.embedding(features)
    return embeddings, targets


def filter_data_by_labels(X, y, labels_to_filter):
    mask = np.isin(y, labels_to_filter)
    filtered_y = y[mask]
    if len(X.shape) == 2:
        filtered_X = X[mask, :]
    elif len(X.shape) == 3:
        filtered_X = X[mask, :, :]
    else:
        raise("Not valid Dimensions")

    return filtered_X, filtered_y

