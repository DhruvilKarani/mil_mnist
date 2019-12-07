import torchvision
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.functional as F
import torchvision.models as pretrained_models
import numpy as np 
import bags
from bags import  ids_by_label ,make_bags, get_patches, num_labels_in_bag
from datetime import datetime
import matplotlib.pyplot as plt


class MILClassifier(nn.Module):
    def __init__(self, bag_size, label_size):
        super(MILClassifier, self).__init__()
        
        self.bag_size = bag_size
        self.label_size = label_size
        self.Conv2d_one = nn.Conv2d(1, 20, kernel_size=5)
        self.Conv2d_two = nn.Conv2d(20, 50, kernel_size=5)
        self.Linear_one = nn.Linear(50 * 4 * 4, 500)    
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.Relu = nn.ReLU()
        self.Softmax = nn.Softmax()
        
    def forward(self, image):
        dim = image.shape[1]
        image = image.view(-1,1,dim,dim).float()
        output = self.Conv2d_one(image)
        output = self.Relu(output)
        output = self.pooling(output)
        output = self.Conv2d_two(output)
        output = self.Relu(output)
        output = self.pooling(output)
        output = output.view(-1,50*4*4)
        output = self.Linear_one(output)
        output = self.Relu(output)
        return output


class Attention(nn.Module):
    def __init__(self,hidden_dim,v_dim):
        super(Attention, self).__init__()
        self.V = nn.Linear(hidden_dim,v_dim,bias=False)
        self.w = nn.Linear(v_dim,1,bias=False)
        
    def forward(self,embeddings):
        weights = torch.zeros(len(embeddings)).to(device)
        norm_factor = 0
        attn_embedding = torch.zeros_like(embeddings[0])
        for i,embedding in enumerate(embeddings):
            embedding = torch.tanh(self.V(embedding))
            embedding = self.w(embedding)
            embedding = torch.exp(embedding)
            norm_factor+=embedding
            weights[i] = embedding
        normalized_weights =  weights.view(-1)/norm_factor.item()
        for weight,embedding in zip(normalized_weights,embeddings):
            attn_embedding+=weight*embedding
        return attn_embedding, normalized_weights


class Classifier(nn.Module):
    def __init__(self,hidden_dim,num_classes):
        super(Classifier, self).__init__()
        self.classify_one = nn.Linear(hidden_dim,200)
        self.classify_two = nn.Linear(200,52)
        self.classify_three = nn.Linear(52, num_classes)
        self.relu = nn.ReLU()
    def forward(self,attn_embedding):
        output = self.classify_one(attn_embedding)
        output = self.relu(output)
        output = self.classify_two(output)
        output = self.relu(output)
        output = self.classify_three(output)
        return output


if __name__ == '__main__':
    
    mnist = torchvision.datasets.MNIST(root='',download=True)
    images = mnist.data #shape = (60000,28,28)
    labels = mnist.targets #shape = (60000)
    ids = np.arange(len(labels))

    today = datetime.today()
    day = today.day
    month = today.month
    file_prefix = str(day)+'_'+str(month) + '_'


    BAG_SIZE = 300
    PROB = 0.6
    NUM_BAGS = 15000
    HIDDEN_DIM = 500
    V_DIM = 128
    NUM_CLASSES = 2
    TARGET = 9
    LOAD_SAVED = False


    target_ids, non_target_ids = ids_by_label(ids,labels,9,'all', True)

    ############################# TESTING ##################################
    for _id in target_ids:
        if labels[_id].item() != TARGET:
            raise ValueError("id: {} is not target".format(labels[_id]))

    for _id in non_target_ids:
        if labels[_id].item() == TARGET:
            raise ValueError("id: {} is equal to target".format(labels[_id]))
    #########################################################################
    
    bag_gen = make_bags(target_ids, non_target_ids, labels, BAG_SIZE, NUM_BAGS, PROB, TARGET)
    sample_bag, sample_label = next(bag_gen)
    patch_set = get_patches(sample_bag,images)


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # sample_batch = images[:BAG_SIZE].to(device)
    # sample_label = torch.LongTensor([1]).to(device).view(-1)

    patch_model = MILClassifier(BAG_SIZE, 2).to(device).train()
    # sample_output = patch_model(sample_batch)
    print(patch_model)

    attention_layer = Attention(HIDDEN_DIM, V_DIM).to(device).train()
    # attention_output, weights = attention_layer(sample_output)
    print(attention_layer)
    classifier = Classifier(HIDDEN_DIM,NUM_CLASSES).to(device).train()
    # output = classifier(attention_output).view(-1,2)
    print(classifier)
    if LOAD_SAVED:
        patch_model.load_state_dict(torch.load('C:/Users/Dhruvil/Desktop/Projects/mil_mnist/models/'+file_prefix+'patch.pth'))
        attention_layer.load_state_dict(torch.load('C:/Users/Dhruvil/Desktop/Projects/mil_mnist/models/'+file_prefix+'attention.pth'))
        classifier.load_state_dict(torch.load('C:/Users/Dhruvil/Desktop/Projects/mil_mnist/models/'+file_prefix+'classifier.pth'))

#---------------------------------------------------- Training Loop -------------------------------------------------------------- 
    class_weights = torch.Tensor([0.5,0.5]).view(NUM_CLASSES).to(device)
    loss_function = nn.CrossEntropyLoss(class_weights)
    patch_optim = optim.Adam(patch_model.parameters(), lr=0.001)
    attn_optim = optim.Adam(attention_layer.parameters(), lr = 0.001)
    class_optim = optim.Adam(classifier.parameters(), lr = 0.001)

    NUM_EPOCHS = 1
    weight_log = []
    loss_log = []
    avg_loss_log = []
    ids_log = []
    bags_log = []
    labels_log = []
    num_targets = []
    for i,(bag,label) in enumerate(bag_gen):
        if torch.cuda.max_memory_allocated(device=None)/1024**3 >4.2 or i == 2000:
            break
        patch_set = get_patches(bag, images).to(device)
        label = torch.LongTensor(label).to(device)

        for _id in bag:
            if label.item() == 0:
                assert labels[_id] != TARGET
                

        patch_model.zero_grad()
        attention_layer.zero_grad()
        classifier.zero_grad()

        patch_output = patch_model(patch_set)
        attention_output, weights = attention_layer(patch_output)
        output = classifier(attention_output).view(-1,2)
        loss = loss_function(output,label)

        loss.backward()
        loss_log.append(loss.item())
        
        class_optim.step()
        attn_optim.step()
        patch_optim.step()

        patch_set.to('cpu')
        torch.cuda.empty_cache()

        if len(loss_log) == 100:
            weight_log.append(weights)
            labels_log.append(label.item())
            ids_log.append(bag)
            avg_loss_log.append(np.mean(loss_log))
            print("--------{}--------".format(i+1))
            print("Average Loss:",avg_loss_log[-1])
            print("Max Weight:",torch.max(weights))
            print("Max Weight Index:",torch.argmax(weights))
            print("CUDA Memory: ",round(torch.cuda.max_memory_allocated(device=None)/1024**3,3))
            loss_log = []

#--------------------------------------------------------------Test Accuracy-----------------------------------------------------------
    patch_model.eval()
    attention_layer.eval()
    classifier.eval()
    tp = 0
    p = 0
    tn = 0
    n = 0
    SAMPLE_NUM = 100
    for i,(bag,label) in enumerate(bag_gen):
            if i == SAMPLE_NUM:
                break
            patch_set = get_patches(bag, images).to(device)
            label = torch.LongTensor(label).to(device)

            for _id in bag:
                if label.item() == 0:
                    assert labels[_id] != TARGET
                    
            patch_output = patch_model(patch_set)
            attention_output, weights = attention_layer(patch_output)
            output = classifier(attention_output).view(-1,2)
            decision = torch.argmax(output)
            print(decision.item(),label.item())
            if label.item() == 0:
                n+=1
                if decision.item() == label.item():
                    tn += 1
            if label.item() == 1:
                p+=1
                if decision.item() == label.item():
                    tp += 1


#-------------------------------------------------------------------plots--------------------------------------------------------------

#save avg loss vs number of images
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(range(len(avg_loss_log)),avg_loss_log, color='r')
    ax.grid(True)
    plt.xlabel("Number of Batches")
    plt.ylabel("Batch Loss")
    plt.legend(['Embedding Dim = {}'.format(HIDDEN_DIM)])
    fig.savefig('C:/Users/Dhruvil/Desktop/Projects/mil_mnist/plots/loss.png')

    
#---------------------------------------------------------------------Logs-------------------------------------------------------------

    with open('C:/Users/Dhruvil/Desktop/Projects/mil_mnist/logs/'+file_prefix+'weights.txt','a+') as file:
        assert len(weight_log) == len(labels_log), "Weight log length = {0}; Labels log length = {1}".format(len(weight_log),len(labels_log))
        for weight,label,ids in zip(weight_log,labels_log,ids_log):
            weight = weight.cpu().detach()
            _id = ids[torch.argmax(weight).item()]
            file.write(str(labels[_id].item()) + "\t" + str(label) + "\t" + str(_id) + "\t" + str(torch.max(weight).item()) + "\t" + str(num_labels_in_bag(ids,labels, TARGET)))
            file.write('\n') 
        file.close
    
    with open('C:/Users/Dhruvil/Desktop/Projects/mil_mnist/logs/'+file_prefix+'loss.txt','a+') as file:
        for loss in avg_loss_log:
            file.write(str(loss))
            file.write('\n') 
        file.close

    with open('C:/Users/Dhruvil/Desktop/Projects/mil_mnist/logs/'+file_prefix+'metrics.txt','a+') as file:
        file.write("\n")
        file.write("P: {}   ".format(p))
        file.write("TP: {}   ".format(tp))
        file.write("N: {}   ".format(n))
        file.write("TN: {}   ".format(tn))
        file.write("\n")
        file.close
#-------------------------------------------------------------Save artifacts------------------------------------------------------------------
    

    torch.save(patch_model.state_dict(), 'C:/Users/Dhruvil/Desktop/Projects/mil_mnist/models/'+file_prefix+'patch.pth')
    torch.save(attention_layer.state_dict(), 'C:/Users/Dhruvil/Desktop/Projects/mil_mnist/models/'+file_prefix+'attention.pth')
    torch.save(classifier.state_dict(), 'C:/Users/Dhruvil/Desktop/Projects/mil_mnist/models/'+file_prefix+'classifier.pth')
