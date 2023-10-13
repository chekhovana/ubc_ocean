from catalyst.contrib.scripts.run import parse_args
import argparse
import sys
import yaml
from omegaconf import OmegaConf
import hydra
from catalyst.dl import (
    BackwardCallback, OptimizerCallback, SchedulerCallback,
    CheckpointCallback, ConsoleLogger, SupervisedRunner
)
from catalyst.custom.loggers.comet import CometLogger
# import os, sys
# import torch
from torch import optim
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import precision_score, balanced_accuracy_score
import warnings

class Trainer:
    def __init__(self, config):
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = hydra.utils.instantiate(config['model'])
        self.model.to(self.device)
        data = hydra.utils.instantiate(config['data'])
        self.loaders = data['loaders']
        # self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4,
        #                        betas=(0.9, 0.999), weight_decay=10e-5)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()


    # def process_train_results(self, results):
    #     loss = 0
    #     predictions, labels = [], []
    #     for result in results:
    #         loss += result['loss'].item()
    #         predictions.append(int(torch.argmax(result['outputs']).detach().cpu().item() > 0.5))
    #         labels.append(result['targets'].detach().cpu().item())
    #     loss /= len(results)
    #     precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    #     score = balanced_accuracy_score(labels, predictions)
    #     return dict(loss=loss, score=score, precision=precision)

    def process_train_results(self, results):
        loss = 0
        predictions, labels = [], []
        for result in results:
            loss += result['loss'].item()
            predictions.append(torch.argmax(result['outputs']).detach().cpu().item())
            labels.append(torch.argmax(result['targets']).detach().cpu().item())
        loss /= len(results)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        score = balanced_accuracy_score(labels, predictions)
        return dict(loss=loss, score=score, precision=precision)

    def train_epoch(self, epoch):
        self.model.train()
        loader = self.loaders['train']
        results = []
        pbar = tqdm(loader)
        for batch in pbar:
            self.optimizer.zero_grad()
            features = batch['features'].squeeze(0).to(self.device)
            targets = batch['targets'].squeeze(0).to(self.device)
            # targets = targets[:1]
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            results.append(dict(loss=loss, outputs=outputs, targets=targets))
            self.optimizer.step()
            result = self.process_train_results(results)
            pbar.set_postfix(result)
        result = self.process_train_results(results)

    def train(self):
        epochs = 50
        for epoch in range(epochs):
            self.train_epoch(epoch)
    # callbacks = [
    #     MultiheadCriterionCallback(metric_key='loss', input_key='logits',
    #                                target_key='targets'),
    #     BackwardCallback(metric_key='loss'),
    #     OptimizerCallback(metric_key='loss'),
    #     SchedulerCallback(loader_key='valid', metric_key='loss'),
    #     CheckpointCallback(loader_key='valid', metric_key='loss',
    #                        minimize=True, logdir='logs/checkpoints')]
    # callbacks = append_metric_callbacks(callbacks,
    #                                     config['data']['num_classes'])
    # loggers = dict(console=ConsoleLogger(),
    #                comet=CometLogger(project_name='tower_classifier',
    #                                  checkpoint_dir=config['checkpoint_dir'],
    #                                  config_file=config_path)
    #                )
    # runner = SupervisedRunner()
    # runner.train(model=model, loaders=loaders,
    #              criterion=config['criterion'], optimizer=optimizer,
    #              scheduler=scheduler, callbacks=callbacks,
    #              loggers=loggers, logdir=config['logdir'],
    #              num_epochs=config['num_epochs'], verbose=True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    print('config', args.config)
    with open(args.config) as f:
        config = yaml.load(f, yaml.Loader)
        trainer = Trainer(config)
        trainer.train()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
