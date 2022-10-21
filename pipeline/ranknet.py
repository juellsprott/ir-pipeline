import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import collections

  
def get_pair_passage_data(q_id, x, y):
    # print(q_id)
    # print(x, y)
    pairs = []
    tmp_pair1 = []
    tmp_pair2 = []
    
    for i in range(0, len(q_id) - 1):
        for j in range(i + 1, len(q_id)):         
            if q_id[i] != q_id[j]:
                break

            if (q_id[i] == q_id[j]) and (y[i] != y[j]):

                if y[i] > y[j]:
                    pairs.append([i,j])
                    tmp_pair1.append(x[i])
                    tmp_pair2.append(x[j])
                    
                else:
                    pairs.append([j,i])
                    tmp_pair1.append(x[j])
                    tmp_pair2.append(x[i])
    
    tensor_pair1 = torch.tensor(tmp_pair1)
    tensor_pair2 = torch.tensor(tmp_pair2)
    
    print('found {} passage pairs'.format(len(pairs)))
    return len(pairs), tensor_pair1, tensor_pair2


class Dataset(data.Dataset):

    def __init__(self, args, q_id, x, y):
        
        self.pair_num, self.tensor_pair1, self.tensor_pair2 = get_pair_passage_data(q_id, x, y)

    def __getitem__(self, index):
        return self.tensor_pair1[index], self.tensor_pair2[index]

    def __len__(self):
        return self.pair_num


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            # layer-1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            # layer-2
            nn.Linear(hidden_size1,hidden_size2),
            nn.ReLU(),
            # layer-out
            nn.Linear(hidden_size2, output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        result_1 = self.model(input_1) 
        result_2 = self.model(input_2) 
        pred = self.sigmoid(result_1 - result_2) 
        return pred

    def predict(self, input):
        result = self.model(input)
        return result


def train(args, q_id, x, y):
    
    model = RankNet(args.input_size, args.hidden_size1, args.hidden_size2, args.output_size).to(device)
 
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = args.lr)
    
    dataset = Dataset(args, q_id, x, y)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers=1
    )
    total_step = len(data_loader)

    for epoch in range(args.epochs):
        for i, (pair1, pair2) in enumerate(data_loader):
            pair1 = pair1.to(device)
            pair2 = pair2.to(device)
            label_size = pair1.size()[0]
            pred = model(pair1, pair2)
            loss = criterion(pred, torch.from_numpy(np.ones(shape=(label_size, 1))).float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), 'ranknet.ckpt')
    print("Save checkpoint {}".format('ranknet.ckpt'))


def inference(args, q_id, p_id, x):

    model = RankNet(args.input_size, args.hidden_size1, args.hidden_size2, args.output_size).to(device)
    
    print("Load checkpoint {}".format('ranknet.ckpt'))
    model.load_state_dict(torch.load('ranknet.ckpt'))
    model.eval()
    
    with torch.no_grad():
        tensor_x = torch.tensor(x).to(device)
        y = model.predict(tensor_x)  
    
    scores = collections.defaultdict(dict)

   
    
    for combination in zip(q_id,p_id,y):
        scores[combination[0]][combination[1]]=combination[2].item()

    return scores


        
