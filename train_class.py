import copy
import torch
import torchvision
import dataset
import mlflow

class TrainBox:
    def __init__(
            self,
            model = None,
            optimizer = None,
            criterion = None,
            scheduler = None,
            train_data: dataset.Dataset = None,
            val_data: dataset.Dataset = None,
            epochs = 100,
            mlflow_log:bool=True
    ):
        self._model=model
        self._best_model=model
        self._optimizer=optimizer
        self._criterion=criterion
        self._scheduler=scheduler

        self.epochs=epochs
        
        self.train_data=train_data
        self.val_data=val_data
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
        #train/val are split by image file beforehand (dataset.split_by_file), so no image is in both
        self._trainloader=torch.utils.data.DataLoader(self.train_data,batch_size=4,shuffle=True)
        #validation uses whole 1024x1024 images, one at a time, same pixels every epoch
        self._validloader=torch.utils.data.DataLoader(self.val_data,batch_size=1,shuffle=False)

    def train(self):
        self._model.to(self._device)
        #check if dataloaders are valid
        if self._trainloader == None or self._validloader==None: #or self._testloader==None:
            raise RuntimeError("Data has not been loaded.")
        #training loop
        print(f"Will run for {self.epochs} epoch(s).")
        best_val_loss=float('inf')
        best_val_iou=-1.0   #below any real IoU, so the first epoch is always saved
        for epoch in range(self.epochs):
            self._model.train()
            train_loss=0
            cur_lr=self._optimizer.param_groups[0]['lr']
            #training
            for images,masks in self._trainloader:
                images,masks=images.to(self._device),masks.to(self._device)

                self._optimizer.zero_grad()
                outputs=self._model(images)

                loss=self._criterion(outputs,masks)
                loss.backward()
                self._optimizer.step()

                train_loss+=loss.item()
            train_loss/=len(self._trainloader)

            #validation
            self._model.eval()
            val_loss=0
            intersection=0
            union=0

            with torch.no_grad():
                for images,masks in self._validloader:
                    images,masks=images.to(self._device),masks.to(self._device)
                    outputs=self._model(images)
                    loss=self._criterion(outputs,masks)
                    val_loss+=loss.item()
                    #pixel IoU over the whole validation set (prediction = probability > 0.5)
                    preds=outputs>0.5
                    targets=masks>0.5
                    intersection+=(preds&targets).sum().item()
                    union+=(preds|targets).sum().item()

                val_loss/=len(self._validloader)
                val_iou=intersection/max(union,1)

            
            #pass validation loss to scheduler
            if self._scheduler is not None:
                self._scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}")
            #save model with improving validation IoU
            if val_iou>best_val_iou:
                self._best_model=copy.deepcopy(self._model)
                best_val_loss=val_loss
                best_val_iou=val_iou
                torch.save({
                    "epoch":epoch,
                    "model_state_dict":self._model.state_dict(),
                    "optimizer_state_dict":self._optimizer.state_dict(),
                    "val_loss":val_loss,
                    "val_iou":val_iou,
                    "scheduler_state_dict":self._scheduler.state_dict() if self._scheduler is not None else None
                },"best_model.pt")
                print(f"Model saved")
            
            #mlflow logging
            mlflow.log_metric("train_loss",train_loss,step=epoch)
            mlflow.log_metric("valid_loss",val_loss,step=epoch)
            mlflow.log_metric("valid_iou",val_iou,step=epoch)
            mlflow.log_metric("cur_lr",cur_lr,step=epoch)
        print(f"Training done. Best validation IoU: {best_val_iou:.4f} (validation loss {best_val_loss:.4f})")
        mlflow.log_metric("best_valid_iou", best_val_iou)

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
