import copy
import os
import random
import time
import numpy as np
import torch
import torchvision
import dataset
import tiling
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
            mlflow_log:bool=True,
            run_config:dict=None,
            val_tile_size=None,
            tile_overlap=128
    ):
        self._model=model
        self._best_model=model
        self._optimizer=optimizer
        self._criterion=criterion
        self._scheduler=scheduler

        self.epochs=epochs
        self.mlflow_log=mlflow_log  #False = train without logging anything to MLflow
        self.checkpoint_path=checkpoint_path  #where the best model of this run is saved
        #latest state, rewritten every epoch so an interrupted run can be resumed (main.py --resume)
        self.last_checkpoint_path=os.path.splitext(checkpoint_path)[0]+"_last.pt"
        self.run_config=run_config  #main.py settings, stored in the checkpoint so a resume rebuilds the same run
        self._start_epoch=0         #>0 after load_state(): continue from there
        self._resume_state=None     #best IoU/loss and early-stopping counter restored by load_state()
        self.early_stopping_patience=early_stopping_patience  #stop after this many epochs without a better val IoU (None = never)
        self.num_workers=num_workers  #background processes preparing batches while the GPU trains (0 = main process only)
        self.val_tile_size=val_tile_size  #validate in overlapping tiles of this size (None = whole image at once)
        self.tile_overlap=tile_overlap    #pixels shared by neighbouring validation tiles
        
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
        #validation uses whole images (in overlapping tiles if val_tile_size is set), one at a time, same pixels every epoch
        self._validloader=torch.utils.data.DataLoader(self.val_data,batch_size=1,shuffle=False,**workers)

    def _log_metric(self,key,value,step=None):
        if self.mlflow_log:
            mlflow.log_metric(key,value,step=step)

    def _save_last(self,epoch,best_val_iou,best_val_loss,epochs_without_improvement):
        """Full training state after `epoch`. Written to a temp file first, then renamed, so a crash
        mid-save never leaves a half-written checkpoint."""
        active=mlflow.active_run() if self.mlflow_log else None
        state={
            "epoch":epoch,
            "model_state_dict":self._model.state_dict(),
            "optimizer_state_dict":self._optimizer.state_dict(),
            "scheduler_state_dict":self._scheduler.state_dict() if self._scheduler is not None else None,
            "best_model_state_dict":self._best_model.state_dict(),
            "best_val_iou":best_val_iou,
            "best_val_loss":best_val_loss,
            "epochs_without_improvement":epochs_without_improvement,
            "rng_torch":torch.get_rng_state(),
            "rng_python":random.getstate(),
            "rng_numpy":np.random.get_state(),
            "config":self.run_config,
            "best_checkpoint_path":self.checkpoint_path,
            "mlflow_run_id":active.info.run_id if active is not None else None,
        }
        os.makedirs(os.path.dirname(self.last_checkpoint_path) or ".",exist_ok=True)
        tmp=self.last_checkpoint_path+".tmp"
        torch.save(state,tmp)
        os.replace(tmp,self.last_checkpoint_path)

    def load_state(self,state):
        """Restore from a checkpoint dict: a `_last.pt` (full state) or an older best-model checkpoint
        (no early-stopping counter or RNG state: those start fresh)."""
        self._model.load_state_dict(state["model_state_dict"])
        self._model.to(self._device)
        self._optimizer.load_state_dict(state["optimizer_state_dict"])
        if self._scheduler is not None and state.get("scheduler_state_dict") is not None:
            self._scheduler.load_state_dict(state["scheduler_state_dict"])
        self._best_model=copy.deepcopy(self._model)
        if state.get("best_model_state_dict") is not None:
            self._best_model.load_state_dict(state["best_model_state_dict"])
        if "rng_torch" in state:
            torch.set_rng_state(state["rng_torch"])
            random.setstate(state["rng_python"])
            np.random.set_state(state["rng_numpy"])
        self._start_epoch=state["epoch"]+1
        self._resume_state={
            "best_val_iou":state.get("best_val_iou",state.get("val_iou",-1.0)),
            "best_val_loss":state.get("best_val_loss",state.get("val_loss",float('inf'))),
            "epochs_without_improvement":state.get("epochs_without_improvement",0),
        }
        print(f"Resuming after epoch {self._start_epoch} (best validation IoU so far {self._resume_state['best_val_iou']:.4f})")

    def train(self):
        self._model.to(self._device)
        #check if dataloaders are valid
        if self._trainloader == None or self._validloader==None: #or self._testloader==None:
            raise RuntimeError("Data has not been loaded.")
        if self.mlflow_log and self._resume_state is None:   #a resumed run already has its params
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
        print(f"Will run epochs {self._start_epoch+1}-{self.epochs}.")
        best_val_loss=float('inf')
        epochs_without_improvement=0
        epoch_times=[]
        training_start=time.perf_counter()
        best_val_iou=-1.0   #below any real IoU, so the first epoch is always saved
        if self._resume_state is not None:
            best_val_iou=self._resume_state["best_val_iou"]
            best_val_loss=self._resume_state["best_val_loss"]
            epochs_without_improvement=self._resume_state["epochs_without_improvement"]
        for epoch in range(self._start_epoch,self.epochs):
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

            prev_images,prev_outputs=None,None
            with torch.no_grad():
                for images,masks in self._validloader:
                    images,masks=images.to(self._device),masks.to(self._device)
                    #annotator copies of the same image come one after another: predict once, score every copy
                    if prev_images is not None and torch.equal(images,prev_images):
                        outputs=prev_outputs
                    else:
                        outputs=tiling.predict_tiled(self._model,images,self.val_tile_size,self.tile_overlap)
                        prev_images,prev_outputs=images,outputs
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

            #update the learning rate: plateau scheduling needs the validation loss, epoch-based schedules (cosine) do not
            if isinstance(self._scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                self._scheduler.step(val_loss)
            elif self._scheduler is not None:
                self._scheduler.step()

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
                    "scheduler_state_dict":self._scheduler.state_dict() if self._scheduler is not None else None,
                    "config":self.run_config,   #lets evaluate.py rebuild the right architecture
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

            #resume point: written every epoch, after logging, so a restart continues from the next epoch
            self._save_last(epoch,best_val_iou,best_val_loss,epochs_without_improvement)

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
