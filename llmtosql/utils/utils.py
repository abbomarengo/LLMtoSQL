import pickle
import os
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def load_history(file_dir):
    path = os.path.join(file_dir, "history.pkl")
    history = pickle.load(open(path, "rb"))
    return history


def load_model(model, PATH):
    # If model was trained in parallel
    try:
        model.load_state_dict(torch.load(PATH))
    except Exception:
        state_dict = torch.load(PATH)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    except Exception as e:
        raise AssertionError(e)
    return model


def plot_history_base(history: dict) -> None:
    x = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    if history['metric_type'] != None:
        train_metric = history['train_metric']
        val_metric = history['val_metric']
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(x, train_loss, c='C0', label='train')
        axes[0].plot(x, val_loss, c='C1', label='validation')
        axes[1].plot(x, train_metric, c='C0', label='train')
        axes[1].plot(x, val_metric, c='C1', label='validation')
        if len(x) > 25:
            axes[0].set_xticks(np.arange(0, len(x)+1, 5))
            axes[0].set_xticklabels(np.arange(0, len(x)+1, 5), rotation=45)
            axes[1].set_xticks(np.arange(0, len(x)+1, 5))
            axes[1].set_xticklabels(np.arange(0, len(x)+1, 5), rotation=45)
        else:
            axes[0].set_xticks(x)
            axes[1].set_xticks(x)
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel(history['metric_type'])
        axes[0].set_title('Training Loss vs. Validation Loss')
        axes[1].set_title(f"{history['metric_type']} - Training vs. Validation")
        axes[0].legend()
        axes[1].legend()
    else:
        plt.subplots(figsize=(10, 5))
        plt.plot(x, train_loss, c='C0', label='train')
        plt.plot(x, val_loss, c='C1', label='validation')
        plt.xticks(x, rotation=45)
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Validation Loss')
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history(history: dict) -> None:
    x = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    if history['metric_type'] != None:
        check = all(isinstance(item, list) for item in history['train_metric']) & \
                all(isinstance(item, list) for item in history['val_metric'])
        if check:
            train_metrics = list(map(list, zip(*history['train_metric'])))
            val_metrics = list(map(list, zip(*history['val_metric'])))
            title = ["Training Loss vs. Validation Loss"]
            sub_titles = [f"{history['metric_type']}_{idx} - Training vs. Validation"
                          for idx in range(len(train_metrics))]
            title.extend(sub_titles)
            fig = make_subplots(
                rows=2, cols=len(train_metrics),
                specs=[[{"colspan": 2}, None],
                       [{}, {}]],
                subplot_titles=title)
        else:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Training Loss vs. Validation Loss",
                                f"{history['metric_type']} - Training vs. Validation"))
        fig.add_trace(go.Scatter(x=x, y=train_loss, name='Train Loss',
                                 line = dict(color='black', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=val_loss, name='Validation Loss',
                                 line=dict(color='black', width=3, dash='dash')), row=1, col=1)
        fig.update_xaxes(title_text='Epochs', row=1, col=1)
        fig.update_yaxes(title_text='Loss', row=1, col=1)
        if check:
            for idx, (train_metric, val_metric) in enumerate(zip(train_metrics, val_metrics), start=1):
                fig.add_trace(go.Scatter(x=x, y=train_metric, name=f"Train {history['metric_type']}",
                                         line=dict(color='blue', width=3)), row=2, col=idx)
                fig.add_trace(go.Scatter(x=x, y=val_metric, name=f"Validation {history['metric_type']}",
                                         line=dict(color='blue', width=3, dash='dash')), row=2, col=idx)
                fig.update_xaxes(title_text='Epochs', row=2, col=idx)
                fig.update_yaxes(title_text='Accuracy', row=2, col=idx)
        else:
            fig.add_trace(go.Scatter(x=x, y=history['train_metric'], name=f"Train {history['metric_type']}",
                                     line=dict(color='blue', width=3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=history['val_metric'], name=f"Validation {history['metric_type']}",
                                     line=dict(color='blue', width=3, dash='dash')), row=2, col=1)
            fig.update_xaxes(title_text='Epochs', row=2, col=1)
            fig.update_yaxes(title_text='Accuracy', row=2, col=1)
        fig.update_layout(showlegend=True, title_text="Model Training Results", hovermode='x unified')
        fig.update_layout(legend=dict(y=0.5, font_size=16), width=1200, height=800)
        fig.show()
