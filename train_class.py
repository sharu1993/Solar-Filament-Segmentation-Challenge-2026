import torch
import torchvision
import dataset
import mlflow
import pdb

class TrainBox:
    def __init__(
            self,
            model = None,
            optimizer = None,
            criterion = None,
            scheduler = None,
            data: dataset.Dataset = None,
            epochs = 100,
            mlflow_log:bool=True
    ):
        self._model=model
        self._best_model=model
        self._optimizer=optimizer
        self._criterion=criterion
        self._scheduler=scheduler

        self.epochs=epochs
        
        self.data=data
        self._trainloader=None
        self._validloader=None
        self._testloader=None

        
        if torch.backends.mps.is_available():
            self._device=torch.device("mps")
            print("Using MPS")
        elif torch.cuda.is_available():
            self._device=torch.device("cuda")
            print("Using CUDA")
        else:
            self._device=torch.device("cpu")
            print("Using CPU")


    def prep_data_loaders(
            self
    ):
        train_size=int(0.8*len(self.data))
        val_size=len(self.data)-train_size
        train_data, val_data=torch.utils.data.random_split(self.data,[train_size,val_size])
        #dataloaders
        self._trainloader=torch.utils.data.DataLoader(train_data,batch_size=16,shuffle=True)
        self._validloader=torch.utils.data.DataLoader(val_data,batch_size=16,shuffle=False)

    def train(self):
        self._model.to(self._device)
        #check if dataloaders are valid
        if self._trainloader == None or self._validloader==None: #or self._testloader==None:
            raise RuntimeError("Data has not been loaded.")
        #training loop
        print(f"Will run for {self.epochs} epoch(s).")
        for epoch in range(self.epochs):
            self._model.train()
            train_loss=0
            cur_lr=self._optimizer.param_groups[0]['lr']
            #training
            for images,masks in self._trainloader:
                pdb.set_trace()
                images,masks=images.to(self._device),masks.to(self._device)

                self._optimizer.zero_grad()
                outputs=self._model(images)

                loss=self._criterion(outputs,masks)
                loss.backward()
                self._optimizer.step()

                train_loss+=loss
            train_loss/=len(self._trainloader)

            #validation
            self._model.eval()
            val_loss=0

            best_val_loss=float('inf')
            best_val_acc=float('inf')
            
            with torch.no_grad():
                for images,masks in self._validloader:
                    images,masks=images.to(self._device),masks.to(self._device)
                    outputs=self._model(images)
                    loss=self._criterion(outputs,masks)
                    val_loss+=loss

                val_loss/=len(self._validloader)

            
            #pass validation loss to scheduler
            self._scheduler.step(val_loss)

            print(f"Epoch {epoch}/{self.epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc:.4f}")
            #save model with improving validation loss
            if best_val_loss>val_loss:
                self._best_model=self._model
                best_val_loss=val_loss
                torch.save({
                    "epoch":epoch,
                    "model_state_dict":self._model.state_dict(),
                    "optimizer_state_dict":self._optimizer.state_dict(),
                    "val_loss":val_loss,
                    "scheduler_state_dict":self._scheduler.state_dict()
                },"best_model.pt")
                print(f"Model saved")
            
            #mlflow logging
            mlflow.log_metric("train_loss",train_loss,step=epoch)
            mlflow.log_metric("valid_loss",val_loss,step=epoch)
            mlflow.log_metric("cur_lr",cur_lr,step=epoch)
        print(f"Training done")
        mlflow.log_metric("Best Validation Accuracy", best_val_acc)

    def run(self):
        self.prep_data_loaders()
        self.train()
    
    def get_trained_model(self):
        return self._best_model
    
    def eval(self):
        self._best_model.eval()
        self._best_model.to(self._device)
        correct=0
        total=0
        with torch.no_grad():
            for images,labels in self._testloader:
                images,labels=images.to(self._device),labels.to(self._device)
                outputs=self._best_model(images)
                pred=outputs.argmax(dim=1)
                correct+=(pred==labels).sum().item()
                total+=labels.size(0)
        test_acc=correct/total
        print(f"Test Accuracy: {test_acc:.4f}")
        mlflow.log_metric("Test Accuracy",test_acc)
        return test_acc
