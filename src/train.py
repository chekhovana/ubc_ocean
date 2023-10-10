from catalyst.contrib.scripts.run import parse_args
import argparse

import yaml
from commons.classification.multihead import MultiheadCriterionCallback
from omegaconf import OmegaConf
import hydra
from catalyst.dl import (
    BackwardCallback, OptimizerCallback, SchedulerCallback,
    CheckpointCallback, ConsoleLogger, SupervisedRunner
)
from catalyst.custom.loggers.comet import CometLogger
# import os, sys
# import torch
from commons.classification.multihead import append_metric_callbacks


def main(config_path):
    config = OmegaConf.load(config_path)
    config = hydra.utils.instantiate(config)
    model = config['model']
    loaders = config['loaders']
    optimizer = config['optimizer'](params=model.parameters())
    scheduler = config['scheduler'](optimizer=optimizer)
    callbacks = [
        MultiheadCriterionCallback(metric_key='loss', input_key='logits',
                                   target_key='targets'),
        BackwardCallback(metric_key='loss'),
        OptimizerCallback(metric_key='loss'),
        SchedulerCallback(loader_key='valid', metric_key='loss'),
        CheckpointCallback(loader_key='valid', metric_key='loss',
                           minimize=True, logdir='logs/checkpoints')]
    callbacks = append_metric_callbacks(callbacks,
                                        config['data']['num_classes'])
    loggers = dict(console=ConsoleLogger(),
                   comet=CometLogger(project_name='tower_classifier',
                                     checkpoint_dir=config['checkpoint_dir'],
                                     config_file=config_path)
                   )
    runner = SupervisedRunner()
    runner.train(model=model, loaders=loaders,
                 criterion=config['criterion'], optimizer=optimizer,
                 scheduler=scheduler, callbacks=callbacks,
                 loggers=loggers, logdir=config['logdir'],
                 num_epochs=config['num_epochs'], verbose=True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    print('config', args.config)
    main(args.config)
