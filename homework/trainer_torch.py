import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, model, optimizer, metric, loss_fn, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric

        self.train_scores = []
        self.dev_scores = []

        self.train_loss = []
        self.dev_loss = []

    def train(self, train_loader, dev_loader, **kwargs):
        num_epochs = kwargs.get("num_epochs", 10)
        log_epochs = kwargs.get("log_epochs", 1)

        save_dir = kwargs.get("save_dir", None)

        best_score = 0
        
    

        for epoch in range(num_epochs):
            
    
            for batch in dev_loader:
                X_dev,y_dev = batch

            for batch in train_loader:
                loss = []
                X_train, y_train = batch
                
                logits = self.model(X_train)
                trn_loss = self.loss_fn(logits, y_train) # return a tensor
                loss.append(trn_loss.item())

                trn_score = self.metric(logits, y_train)
                self.train_scores.append(trn_score)

                self.optimizer.zero_grad()
                trn_loss.backward()
                self.optimizer.step()
            
            dev_score, dev_loss = self.evaluate(X_dev,y_dev)
            if dev_score > best_score:
                print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
                if save_dir:
                    self.save_model(save_dir)

            if log_epochs and epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}/{num_epochs}, loss: {trn_loss.item()}")
            self.train_loss.append(sum(loss)/len(loss))

    def evaluate(self, X,y):
        #self.model.eval()
        logits = self.model(X)

        loss = self.loss_fn(logits, y).item()
        self.dev_loss.append(loss)
        score = self.metric(logits, y)
        self.dev_scores.append(score)
        return score, loss

    def predict(self, X):
        return self.model(X)

    def save_model(self, save_dir):
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                with open(os.path.join(save_dir, layer.name+".pdparams"),'wb') as fout:
                    pickle.dump(layer.params, fout)

    def load_model(self, model_dir):
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pdparams","")
            name_file_dict[name] = os.path.join(model_dir, file_name)

        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                name = layer.name
                file_path = name_file_dict[name]
                with open(file_path,'rb') as fin:
                    layer.params = pickle.load(fin)

    def plot(self):
        x = range(len(self.train_loss))
        plt.plot(x, self.train_loss, label='train_loss', color='blue')
        plt.plot(x, self.dev_loss, label='dev_loss', color='red')
        plt.legend()
        plt.title('train and dev loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        plt.plot(x,self.dev_scores,label ='dev_score',color = 'blue')
        plt.legend()
        plt.title('dev score')
        plt.xlabel('epoch')
        plt.ylabel('score')
        plt.show()