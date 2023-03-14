import torch
from torch.utils.data import distributed, DataLoader
from torch.nn import parallel, LogSoftmax, Softmax
from torch import optim
from sklearn.metrics import accuracy_score
import structlog
import os
import gc
from tqdm import tqdm
import pickle

# SageMaker data parallel: Import PyTorch's distributed API
import torch.distributed as dist

# Local import

from .utils.functions import custom_loss_function

logger = structlog.get_logger('__name__')


class Trainer():
    def __init__(self, model, datasets=None, epochs=None, batch_size=None, n_metrics=1,
                 is_parallel=False, save_history=False, **config):
        logger.info('Config inputs.', config=config)
        allowed_kwargs = {"seed", "scheduler", "optimizer", "momentum", "weight_decay",
                          "lr", "criterion", "metric", "pred_function", "model_dir", "backend"}
        self.validate = True
        self.n_metrics = n_metrics
        self.validate_kwargs(config, allowed_kwargs)
        # Unpack kwargs
        self.epochs = epochs
        self.scheduler_type = config.get('scheduler', None)
        self.optimizer_type = config.get('optimizer', 'sgd')
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.lr = config.get('lr', 0.001)
        self.criterion_type = config.get('criterion', 'cross_entropy')
        self.metric = config.get('metric', 'accuracy')
        self.pred_function_type = config.get('pred_function', 'softmax')
        self.model_dir = config.get('model_dir', 'model_output')
        backend = config.get('backend', 'smddp')
        seed = config.get('seed', 32)
        # SageMaker data parallel: Import the library PyTorch API
        if is_parallel:
            import smdistributed.dataparallel.torch.torch_smddp
        if datasets:
            if len(datasets) == 2:
                train_set, val_set = datasets
            else:
                train_set = datasets
                self.validate = False
        torch.manual_seed(seed)
        self.model = model
        self.is_parallel = is_parallel
        self.save_history = save_history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.history = {}
        logger.info("Loading the model.")
        if self.is_parallel:
            if datasets:
                dist.init_process_group(backend=backend)
                train_sampler = distributed.DistributedSampler(train_set, num_replicas=dist.get_world_size(),
                                                               rank=dist.get_rank())
                # Scale batch size by world size
                batch_size = batch_size // dist.get_world_size()
                batch_size = max(batch_size, 1)
            else:
                logger.warning("Testing only available. No datasets in arguments.")
        else:
            if datasets:
                train_sampler = None
            else:
                logger.warning("Testing only available. No datasets in arguments.")
        if datasets:
            logger.info('Loading training and validation set.')
            logger.info("Preparing the data.")
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler)
            if self.validate:
                self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
            logger.debug(
                "Processes {}/{} ({:.0f}%) of train data".format(
                    len(self.train_loader.sampler),
                    len(self.train_loader.dataset),
                    100.0 * len(self.train_loader.sampler) / len(self.train_loader.dataset),
                )
            )
            if self.validate:
                logger.debug(
                    "Processes {}/{} ({:.0f}%) of validation data".format(
                        len(self.val_loader.sampler),
                        len(self.val_loader.dataset),
                        100.0 * len(self.val_loader.sampler) / len(self.val_loader.dataset),
                    )
                )
        else:
            logger.warning("Testing only available. No datasets in arguments.")
        if self.is_parallel:
            local_rank = os.environ["LOCAL_RANK"]
            torch.cuda.set_device(int(local_rank))
            # self.model.cuda(int(local_rank))
            cuda = "cuda:"+local_rank
            self.device = torch.device(cuda)
            self.model = self.model.to(self.device)
            self.model = parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        else:
            # if torch.backends.mps.is_available():
            #     self.device = torch.device("mps")
            # else:
            #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
        logger.info(f'Training on device: {self.device}.')
        criterion = self._get_criterion()
        if self.criterion_type != 'custom':
            self.criterion = criterion.to(self.device)
        else:
            self.criterion = criterion
        self.optimizer = self._get_optimizer()
        self.scheduler_options = {
            'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5,
                                                                                                eta_min=1e-7),
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-7),
            'StepLR': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2)
        }
        if self.scheduler_type:
            self.scheduler = self.scheduler_options[self.scheduler_type]
        self.pred_function = self._get_prediction_function()

    def _get_prediction_function(self):
        if self.pred_function_type == 'logsoftmax':
            return LogSoftmax(dim=-1)
        elif self.pred_function_type == 'softmax':
            return Softmax(dim=-1)
        else:
            return None

    def _get_optimizer(self):
        if self.optimizer_type == 'sgd':
            logger.info('Using SGD optimizer')
            return optim.SGD(self.model.parameters(), lr=self.lr,
                             momentum=self.momentum, weight_decay=self.weight_decay)
        if self.optimizer_type == 'adam':
            logger.info('Using ADAM optimizer')
            return optim.Adam(self.model.parameters(), lr=self.lr,
                              weight_decay=self.weight_decay)
        if self.optimizer_type == 'adagrad':
            logger.info('Using ADAGRAD optimizer')
            return optim.Adagrad(self.model.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
        if self.optimizer_type == 'adamax':
            logger.info('Using ADAMAX optimizer')
            return optim.Adamax(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        if self.optimizer_type == 'adamw':
            logger.info('Using ADAMW optimizer')
            return optim.AdamW(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

    def _get_criterion(self):
        if self.criterion_type == 'cross_entropy':
            logger.info('Using CROSS ENTROPY loss')
            return torch.nn.CrossEntropyLoss()
        if self.criterion_type == 'neg-loss':
            logger.info('Using NEG LOSS loss')
            return torch.nn.NLLLoss
        if self.criterion_type == 'l1':
            logger.info('Using L1 loss')
            return torch.nn.L1Loss()
        if self.criterion_type == 'l2':
            logger.info('Using L2 loss')
            return torch.nn.MSELoss
        if self.criterion_type == 'custom':
            logger.info('Using CUSTOM loss')
            return custom_loss_function

    def _average_gradients(self):
        # Average gradients (only for multi-node CPU)
        # Gradient averaging.
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def _evaluate(self, outputs, targets):
        if self.metric == 'mcrmse':
            colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
            return torch.mean(torch.sqrt(colwise_mse), dim=0)
        if self.metric == 'accuracy':
            predictions = self._get_predictions(outputs)
            if isinstance(predictions, tuple) or isinstance(predictions, list):
                return tuple(accuracy_score(target.cpu().detach().numpy(), prediction.cpu().detach().numpy())
                             for target, prediction in zip(targets, predictions))
            else:
                return accuracy_score(targets.cpu().detach().numpy(), predictions.cpu().detach().numpy())

    def _get_predictions(self, outputs):
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            if self.pred_function_type:
                return tuple(torch.argmax(self.pred_function(output), dim=-1) for output in outputs)
            else:
                return tuple(torch.argmax(output, dim=-1) for output in outputs)
        else:
            if self.pred_function_type:
                return torch.argmax(self.pred_function(outputs), dim=-1)
            else:
                return torch.argmax(outputs, dim=-1)

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.model = self.model.to(self.device)
        running_loss = 0.
        if self.n_metrics == 1:
            running_metric = 0.
        else:
            running_metric = [.0] * self.n_metrics
        with tqdm(self.train_loader, unit='batch') as tepoch:
            for i, data in enumerate(tepoch):
                self.optimizer.zero_grad()
                outputs = self.model_forward(data)
                targets = self.gather_targets(data)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                if self.scheduler_type == 'CosineAnnealingWarmRestarts':
                    self.scheduler.step(epoch - 1 + i / len(self.train_loader))  # as per pytorch docs
                if self.metric:
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        if len(outputs) == self.n_metrics:
                            evaluated_metrics = self._evaluate(outputs, targets)
                            for idx, metric in enumerate(running_metric):
                                metric += evaluated_metrics[idx]
                                running_metric[idx] = metric
                            exec(self._create_string())
                        else:
                            raise IndexError('Number of outputs and number of metrics not matching.')
                    else:
                        running_metric += self._evaluate(outputs, targets)
                        tepoch.set_postfix(loss=running_loss / len(self.train_loader),
                                           metric=running_metric / len(self.train_loader))
                else:
                    tepoch.set_postfix(loss=loss.item())
                del targets, outputs, loss
        if self.scheduler_type == 'StepLR':
            self.scheduler.step()
        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)
        if self.metric:
            if self.n_metrics == 1:
                self.train_metrics.append(running_metric / len(self.train_loader))
            else:
                self.train_metrics.append([metric / len(self.train_loader) for metric in running_metric])
        del running_metric

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        running_loss = 0.
        if self.n_metrics == 1:
            running_metric = 0.
        else:
            running_metric = [.0] * self.n_metrics
        with tqdm(self.val_loader, unit='batch') as tepoch:
            for data in tepoch:
                outputs = self.model_forward(data)
                targets = self.gather_targets(data)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                if self.metric:
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        if len(outputs) == self.n_metrics:
                            evaluated_metrics = self._evaluate(outputs, targets)
                            for idx, metric in enumerate(running_metric):
                                metric += evaluated_metrics[idx]
                                running_metric[idx] = metric
                            exec(self._create_string(set_type='val'))
                        else:
                            raise IndexError('Number of outputs and number of metrics not matching.')
                    else:
                        running_metric += self._evaluate(outputs, targets)
                        tepoch.set_postfix(loss=running_loss / len(self.val_loader),
                                           metric=running_metric / len(self.val_loader))
                else:
                    tepoch.set_postfix(loss=loss.item())
                del targets, outputs, loss
        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
        if self.metric:
            if self.n_metrics == 1:
                self.val_metrics.append(running_metric / len(self.val_loader))
            else:
                self.val_metrics.append([metric / len(self.val_loader) for metric in running_metric])
        del running_metric

    def save_model(self, model_dir):
        logger.info("Saving the model.")
        path = os.path.join(model_dir, "model.pth")
        torch.save(self.model.cpu().state_dict(), path)

    def save_history_(self, model_dir):
        logger.info("Saving the training history.")
        path = os.path.join(model_dir, "history.pkl")
        with open(path, "wb") as fp:
            pickle.dump(self.history, fp)

    def fit(self):
        logger.info("Start training..")
        for epoch in range(1, self.epochs + 1):
            logger.info(f"{'-' * 30} EPOCH {epoch} / {self.epochs} {'-' * 30}")
            self._train_one_epoch(epoch)
            self.clear()
            if self.validate:
                self._validate_one_epoch()
                self.clear()
            # Save model on master node.
            if self.is_parallel:
                if dist.get_rank() == 0:
                    self.save_model(self.model_dir)
            else:
                self.save_model(self.model_dir)
            if self.metric:
                logger.info(f"train loss: {self.train_losses[-1]} - "
                            f"train {self.metric}: {self.train_metrics[-1]}")
                if self.validate:
                    logger.info(f"valid loss: {self.val_losses[-1]} - "
                                f"valid {self.metric}: {self.val_metrics[-1]}\n")
            else:
                logger.info(f"train loss: {self.train_losses[-1]}")
                if self.validate:
                    logger.info(f"valid loss: {self.val_losses[-1]}\n")
        self.history = {
            'epochs': [*range(1, self.epochs + 1)],
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_metric': self.train_metrics,
            'val_metric': self.val_metrics,
            'metric_type': self.metric
        }
        if self.save_history:
            self.save_history_(self.model_dir)
        logger.info("Training Complete.")

    def test(self, model, test_loader):
        logger.info("Testing..")
        model = model.to(self.device)
        running_loss = 0.
        running_metric = 0.
        with tqdm(test_loader, unit='batch') as tepoch:
            for data in tepoch:
                targets = self.gather_targets(data)
                outputs = self.model_forward(data, model=model)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                if self.metric:
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        if len(outputs) == self.n_metrics:
                            evaluated_metrics = self._evaluate(outputs, targets)
                            for idx, metric in enumerate(running_metric):
                                metric += evaluated_metrics[idx]
                                running_metric[idx] = metric
                            exec(self._create_string(set_type='test'))
                        else:
                            raise IndexError('Number of outputs and number of metrics not matching.')
                    else:
                        running_metric += self._evaluate(outputs, targets)
                        tepoch.set_postfix(loss=running_loss / len(test_loader),
                                           metric=running_metric / len(test_loader))
                else:
                    tepoch.set_postfix(loss=loss.item())
                del targets, outputs, loss
        test_loss = running_loss / len(test_loader)
        if self.metric:
            if self.n_metrics == 1:
                test_metric = running_metric / len(test_loader)
            else:
                test_metric = [metric / len(test_loader) for metric in running_metric]
            return test_loss, test_metric
        return test_loss

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()

    def validate_kwargs(self, kwargs, allowed_kwargs, error_message="Keyword argument not understood:"):
        """Checks that all keyword arguments are in the set of allowed keys."""
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError(error_message, kwarg)

    def model_forward(self, data, model=None):
        inputs = (data['tokenized_inputs']['question'].to(self.device),
                  data['tokenized_inputs']['columns'].to(self.device))
        if model:
            outputs = model(inputs)
        else:
            outputs = self.model(inputs)
        del inputs
        return outputs

    def gather_targets(self, data):
        sel_labels = data['labels']['sel'].to(self.device)
        agg_labels = data['labels']['agg'].to(self.device)
        return sel_labels, agg_labels

    def _create_string(self, set_type='train'):
        string = f'tepoch.set_postfix(loss=running_loss / len(self.{set_type}_loader)'
        for idx in range(self.n_metrics):
            string += f', metric{idx+1}=running_metric[{idx}] / len(self.{set_type}_loader)'
        string += ')'
        return string
