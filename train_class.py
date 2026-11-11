import copy
import os
import time
import torch
import torchvision
import dataset
import mlflow

def fmt_time(seconds):
    """Seconds -> 'm:ss' (or 'h:mm:ss' when over an hour)."""
    m,s=divmod(int(round(seconds)),60)
    h,m=divmod(m,60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

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
            early_stopping_patience = None,
            num_workers = 0,
            checkpoint_path = "best_model.pt",
            mlflow_log:bool=True
    ):
        self._model=model
        self._best_model=model
        self._optimizer=optimizer
        self._criterion=criterion
        self._scheduler=scheduler

        self.epochs=epochs
        self.mlflow_log=mlflow_log  #False = train without logging anything to MLflow
        self.checkpoint_path=checkpoint_path  #where the best model of this run is saved
        self.early_stopping_patience=early_stopping_patience  #stop after this many epochs without a better val IoU (None = never)
        self.num_workers=num_workers  #background processes preparing batches while the GPU trains (0 = main process only)
        
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
        #persistent_workers keeps the worker processes alive between epochs instead of restarting them
        workers=dict(num_workers=self.num_workers, persistent_workers=self.num_workers>0)
        self._trainloader=torch.utils.data.DataLoader(self.train_data,batch_size=4,shuffle=True,**workers)
        #validation uses whole 1024x1024 images, one at a time, same pixels every epoch
        self._validloader=torch.utils.data.DataLoader(self.val_data,batch_size=1,shuffle=False,**workers)

    def _log_metric(self,key,value,step=None):
        if self.mlflow_log:
            mlflow.log_metric(key,value,step=step)

    def train(self):
        self._model.to(self._device)
        #check if dataloaders are valid
        if self._trainloader == None or self._validloader==None: #or self._testloader==None:
            raise RuntimeError("Data has not been loaded.")
        if self.mlflow_log:
            #training settings that live in this class (data/model settings are logged in main.py)
            mlflow.log_params({
                "epochs_max":self.epochs,
                "early_stopping_patience":self.early_stopping_patience,
                "num_workers":self.num_workers,
                "train_batch_size":self._trainloader.batch_size,
                "val_batch_size":self._validloader.batch_size,
                "train_samples":len(self.train_data),
                "val_samples":len(self.val_data),
                "optimizer":type(self._optimizer).__name__,
                "scheduler":type(self._scheduler).__name__ if self._scheduler is not None else None,
                "loss":type(self._criterion).__name__,
                "device":str(self._device),
                "checkpoint_path":self.checkpoint_path,
            })
        #training loop
        print(f"Will run for {self.epochs} epoch(s).")
        best_val_loss=float('inf')
        epochs_without_improvement=0
        epoch_times=[]
        training_start=time.perf_counter()
        best_val_iou=-1.0   #below any real IoU, so the first epoch is always saved
        for epoch in range(self.epochs):
            epoch_start=time.perf_counter()
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
            train_time=time.perf_counter()-epoch_start

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

            
            val_time=time.perf_counter()-epoch_start-train_time

            #pass validation loss to scheduler
            if self._scheduler is not None:
                self._scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}")
            #save model with improving validation IoU
            if val_iou>best_val_iou:
                self._best_model=copy.deepcopy(self._model)
                best_val_loss=val_loss
                best_val_iou=val_iou
                os.makedirs(os.path.dirname(self.checkpoint_path) or ".",exist_ok=True)
                torch.save({
                    "epoch":epoch,
                    "model_state_dict":self._model.state_dict(),
                    "optimizer_state_dict":self._optimizer.state_dict(),
                    "val_loss":val_loss,
                    "val_iou":val_iou,
                    "scheduler_state_dict":self._scheduler.state_dict() if self._scheduler is not None else None
                },self.checkpoint_path)
                print(f"Model saved to {self.checkpoint_path}")
                epochs_without_improvement=0
            else:
                epochs_without_improvement+=1
            
            #mlflow logging (best_valid_iou every epoch, so it is kept even if training is interrupted)
            self._log_metric("train_loss",train_loss,step=epoch)
            self._log_metric("valid_loss",val_loss,step=epoch)
            self._log_metric("valid_iou",val_iou,step=epoch)
            self._log_metric("best_valid_iou",best_val_iou,step=epoch)
            self._log_metric("cur_lr",cur_lr,step=epoch)

            #timing: this epoch (incl. checkpoint saving), running average, and worst-case time left
            epoch_times.append(time.perf_counter()-epoch_start)
            avg=sum(epoch_times)/len(epoch_times)
            print(f"  Time: train {fmt_time(train_time)}, validation {fmt_time(val_time)}, epoch {fmt_time(epoch_times[-1])} | "
                  f"average {fmt_time(avg)}/epoch | at most {fmt_time(avg*(self.epochs-epoch-1))} left")
            self._log_metric("train_time_s",train_time,step=epoch)
            self._log_metric("val_time_s",val_time,step=epoch)
            self._log_metric("epoch_time_s",epoch_times[-1],step=epoch)

            #early stopping: no better validation IoU for `early_stopping_patience` epochs in a row
            if self.early_stopping_patience is not None and epochs_without_improvement>=self.early_stopping_patience:
                print(f"Early stopping: no improvement in validation IoU for {epochs_without_improvement} epochs.")
                break
        print(f"Training done. Best validation IoU: {best_val_iou:.4f} (validation loss {best_val_loss:.4f})")
        print(f"Total time {fmt_time(time.perf_counter()-training_start)} for {len(epoch_times)} epoch(s), "
              f"average {fmt_time(sum(epoch_times)/max(len(epoch_times),1))}/epoch")
        self._log_metric("epochs_run",len(epoch_times))

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
        self._log_metric("Test Accuracy",test_acc)
        return test_acc
