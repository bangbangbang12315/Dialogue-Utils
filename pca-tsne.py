import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import gensim
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from gensim import similarities
TaggededDocument = gensim.models.doc2vec.TaggedDocument
EMBEDDING_SIZE=100
import numpy as np

def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    return acc

def data2vec(src, model_dm):
    X, Y = [], []
    user_cnt = set()
    with open(src,'r') as f_src:
        x = f_src.readline()
        user_data = json.loads(x)
        for user, sentences in tqdm(user_data.items()):
        # for data in tqdm(f_src):
        # for user, sentences in tqdm(f_src):
            # user, sentences = data.strip().split('\t')
            user_cnt.add(user)
            if len(user_cnt) == 1 or len(user_cnt) == 14:
            # if user == '0' or user == '13':
                for sentence in sentences:
                    X.append(model_dm.infer_vector(sentence.split(' ')))
                    Y.append(int(user))
    return np.array(X), np.array(Y)
        
def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=10, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)

    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id, label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')


def plot_scatter(feat, label, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return

if __name__ == "__main__":
    src_dir = "../data/user_test/user_data"
    model_dm = Doc2Vec.load("model/model_dm")
    latents, labels = data2vec(src_dir, model_dm)
    pred_from_latent, emb_from_latent = predict(latents)
    # acc_latent = cal_acc(valY, pred_from_latent)
    # print('The clustering accuracy is:', acc_latent)
    print('The clustering result:')
    plot_scatter(emb_from_latent, labels, savefig='p1_baseline_2.png')
    