import copy
import torch
import torch.nn as nn
from typing import Type
import client
from tqdm.auto import tqdm
import summary_writer
import aggregation_functions
from pathlib import Path
import os

class Server:
    def __init__(self,
                 num_train_clients: int,  # number of training clients
                 num_test_clients: int,  # number of test clients out of total clients
                 model: Type[nn.Module],
                 loss_fn_type: Type[nn.Module],
                 optimizer_type: Type[torch.optim.Optimizer],
                 model_name,
                 experiment_name,
                 device,
                 transform,
                 data_loader_function,
                 lr):
        self.num_train_clients = num_train_clients
        self.num_test_clients = num_test_clients
        self.model = model.to(device)
        self.experiment_info = {"model_name": model_name, "experiment_name": experiment_name}
        self.loss_fn_type = loss_fn_type
        self.optimizer_type = optimizer_type
        self.transform = transform
        self.train_dataloaders, self.validate_dataloaders, self.test_dataloaders = data_loader_function(self.transform)
        self.lr = lr
        self.device = device
        self.client_models = [copy.deepcopy(model).to(device) for _ in range(num_train_clients)]

    def run(self,
            server_epochs,
            client_epochs,
            folder_path):
        next_epoch = self._resume_training(folder_path)
        writer = summary_writer.get_summary_writer(
            experiment_name=self.experiment_info["experiment_name"],
            model_name=self.experiment_info["model_name"]
        )

        for epoch in tqdm(range(next_epoch, server_epochs)):
            train_loss, validate_loss = 0,0
            train_acc, validate_acc = 0,0
            for j in range(self.num_train_clients):
                self.client_models[j].to(self.device)
                optimizer = self.optimizer_type(params=self.client_models[j].parameters(), lr=self.lr)
                # print('starting training')
                results, new_model = client.train_client(
                    model=self.client_models[j],
                    loss_fn = self.loss_fn_type(),
                    optimizer = optimizer,
                    train_dataloader = self.train_dataloaders[j],
                    validate_dataloader = self.validate_dataloaders[j],
                    device = self.device,
                    epochs = client_epochs
                )
                self.client_models[j] = copy.deepcopy(new_model)
                writer.add_scalar(f"train_loss/client-{j+1}", results["train_loss"][-1], epoch)
                writer.add_scalar(f"train_acc/client-{j+1}", results["train_acc"][-1], epoch)
                writer.add_scalar(f"validate_loss/client-{j+1}", results["validate_loss"][-1], epoch)
                writer.add_scalar(f"validate_acc/client-{j+1}", results["validate_acc"][-1], epoch)

                attempts = 0
                while attempts < 30:
                    attempts += 1
                    try:
                        p = folder_path/f"models/client-{j}/"
                        p.mkdir(parents=True, exist_ok=True)
                        file_path = p/f"epoch-{epoch}.pt"
                        torch.save(obj=self.client_models[j].state_dict(), f=file_path)
                    except:
                        continue
                    else:
                        break


                train_loss += results["train_loss"][-1]
                validate_loss += results["validate_loss"][-1]
                train_acc += results["train_acc"][-1]
                validate_acc += results["validate_acc"][-1]

            train_loss /= self.num_train_clients
            validate_loss /= self.num_train_clients
            train_acc /= self.num_train_clients
            validate_acc /= self.num_train_clients

            self.model = aggregation_functions.fed_avg(client_models=self.client_models, train_dataloaders=self.train_dataloaders)
            # self.model.load_state_dict(self.client_models[0].model.state_dict())
            # for j in range(1, self.num_train_clients):
            #     for(key, value) in self.model.state_dict().items():
            #         self.model.state_dict()[key].copy_(self.client_models[j].model.state_dict()[key] + value)
            
            # for (key, value) in self.model.state_dict().items():
            #     self.model.state_dict()[key].copy_(value/self.num_clients)

            for j in range(self.num_train_clients):
                self.client_models[j].load_state_dict(self.model.state_dict())
                self.client_models[j].to(self.device)
                
                
            
            final_loss, final_acc = 0,0
            for i in range(self.num_train_clients):
                temp_loss, temp_acc = client.validate_step(
                    model=self.client_models[i],
                    loss_fn=self.loss_fn_type(),
                    validate_dataloader=self.validate_dataloaders[i],
                    device=self.device
                )
                final_loss += temp_loss
                final_acc += temp_acc
            
            final_loss /= self.num_train_clients
            final_acc /= self.num_train_clients
            writer.add_scalar("final_validate_loss/client", validate_loss, epoch)
            writer.add_scalar("final_validate_acc/client", validate_acc, epoch)
            writer.add_scalar("final_validate_loss/server", final_loss, epoch)
            writer.add_scalar("final_validate_acc/server", final_acc, epoch)

            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"validate_loss: {validate_loss:.4f} | "
                f"validate_acc: {validate_acc:.4f}"
                "\n"
                "Aggregated Model"
                f"Epoch: {epoch+1} | "
                f"validate_loss: {final_loss:.4f} | "
                f"validate_acc: {final_acc:.4f}"
            )
            attempts = 0
            while attempts < 30:
                attempts += 1
                try:
                    p = folder_path/"models/server/"
                    p.mkdir(parents = True, exist_ok = True)
                    p_file = p/f"epoch-{epoch}.pt"
                    torch.save(obj=self.model.state_dict(), f=p_file)
                except:
                    continue
                else:
                    break
            
        writer.close()


    def _resume_training(self, folder_path):
        server_path = folder_path / "models/server"
        if not server_path.exists():
            return 0

        _,_,files = next(os.walk(server_path))
        completed_epochs = len(files)

        model_files = sorted(server_path.glob("epoch-*.pt"), key=lambda x: int(x.stem.split('-')[1]))
        if not model_files:
            return 0

        latest_epoch = int(model_files[-1].stem.split('-')[1])
        self.model.load_state_dict(torch.load(model_files[-1]))

        for j in range(self.num_train_clients):
            self.client_models[j] = copy.deepcopy(self.model)
        return completed_epochs

    # @staticmethod
    # def _save_model_state(model, file_path):
    #     attempts = 0
    #     while attempts < 30:
    #         attempts += 1
    #         try:
    #             file_path.parent.mkdir(parents=True, exist_ok=True)
    #             torch.save(obj=model.state_dict(), f=file_path)
    #         except:
    #             continue
    #         else:
    #             break
