import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import secrets

class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Simple self-attention mechanism
        # x shape: (batch_size, input_dim)
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        return x * weights

class SlimePredictor(nn.Module):
    def __init__(
        self,
        input_dim=17,
        output_dim=5,
        hidden_layers=None,
        activation="relu",
        dropout=0.2,
        use_attention=True,
    ):
        super(SlimePredictor, self).__init__()
        if hidden_layers is None:
            hidden_layers = [64, 128, 64]
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = [int(x) for x in hidden_layers]
        self.activation_name = str(activation).lower()
        self.dropout_p = float(dropout)
        self.use_attention = bool(use_attention)
        
        self.input_bn = nn.BatchNorm1d(self.input_dim)
        
        self.attention = AttentionBlock(self.input_dim) if self.use_attention else None
        
        layers = []
        prev = self.input_dim
        for width in self.hidden_layers:
            layers.append(nn.Linear(prev, width))
            prev = width
        self.hidden_linears = nn.ModuleList(layers)
        self.output_linear = nn.Linear(prev, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_p)
    
    def _act(self, x):
        if self.activation_name == "relu":
            return F.relu(x)
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_name == "tanh":
            return torch.tanh(x)
        if self.activation_name == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.01)
        return F.relu(x)
        
    def forward(self, x):
        x = self.input_bn(x)
        if self.attention is not None:
            x = self.attention(x)
        
        for layer in self.hidden_linears:
            x = self._act(layer(x))
            x = self.dropout(x)
        
        output = self.output_linear(x)
        return output

class ModelManager:
    def __init__(
        self,
        input_dim=17,
        output_dim=5,
        hidden_layers=None,
        activation="relu",
        optimizer_name="adam",
        learning_rate=0.001,
        weight_decay=0.0,
        l1_lambda=0.0,
        dropout=0.2,
        scheduler_name="none",
        scheduler_params=None,
        seed=None,
        deterministic=False,
        use_attention=True,
        shuffle=True,
    ):
        if hidden_layers is None:
            hidden_layers = [64, 128, 64]
        if scheduler_params is None:
            scheduler_params = {}
        self.config = {
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            "hidden_layers": [int(x) for x in hidden_layers],
            "activation": str(activation).lower(),
            "optimizer_name": str(optimizer_name).lower(),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "l1_lambda": float(l1_lambda),
            "dropout": float(dropout),
            "scheduler_name": str(scheduler_name).lower(),
            "scheduler_params": dict(scheduler_params),
            "seed": None if seed is None else int(seed),
            "deterministic": bool(deterministic),
            "use_attention": bool(use_attention),
            "shuffle": bool(shuffle),
        }

        if seed is not None:
            self.set_seed(int(seed), deterministic=bool(deterministic))

        self.model = SlimePredictor(
            input_dim=self.config["input_dim"],
            output_dim=self.config["output_dim"],
            hidden_layers=self.config["hidden_layers"],
            activation=self.config["activation"],
            dropout=self.config["dropout"],
            use_attention=self.config["use_attention"],
        )

        self.optimizer = self._build_optimizer(
            optimizer_name=self.config["optimizer_name"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        self.scheduler = self._build_scheduler(
            scheduler_name=self.config["scheduler_name"],
            scheduler_params=self.config["scheduler_params"],
        )

        self.criterion = nn.MSELoss()
        self.is_trained = False
        self.last_train_metrics = {}
        self.loss_history = []
        self.lr_history = []

    @staticmethod
    def new_seed():
        return int.from_bytes(secrets.token_bytes(4), "big")

    @staticmethod
    def set_seed(seed, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(False)
    
    def _build_optimizer(self, optimizer_name, learning_rate, weight_decay):
        name = str(optimizer_name).lower()
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        if name == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.0)
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def _build_scheduler(self, scheduler_name, scheduler_params):
        name = str(scheduler_name).lower()
        if name == "steplr":
            step_size = int(scheduler_params.get("step_size", 10))
            gamma = float(scheduler_params.get("gamma", 0.5))
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        if name == "exponentiallr":
            gamma = float(scheduler_params.get("gamma", 0.9))
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        return None
        
    def train(self, X_train, y_train, epochs=100):
        self.model.train()
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        self.last_train_metrics = {"epochs": int(epochs)}
        self.loss_history = []
        self.lr_history = []
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            if self.config["shuffle"]:
                perm = torch.randperm(X_tensor.shape[0])
                X_tensor = X_tensor[perm]
                y_tensor = y_tensor[perm]
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            if self.config["l1_lambda"] > 0:
                l1 = torch.tensor(0.0)
                for p in self.model.parameters():
                    l1 = l1 + p.abs().sum()
                loss = loss + self.config["l1_lambda"] * l1
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.loss_history.append(float(loss.item()))
            self.lr_history.append(float(self.optimizer.param_groups[0]["lr"]))
            
        self.is_trained = True
        self.last_train_metrics["final_loss"] = float(loss.item())
        self.last_train_metrics["final_lr"] = float(self.optimizer.param_groups[0]["lr"])
        return float(loss.item())
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
        return predictions.numpy()

    def build_report(
        self,
        include_weights=True,
        max_tensor_elements=5000,
        extra_metrics=None,
        feature_names=None,
        output_names=None,
        output_units=None,
    ):
        if extra_metrics is None:
            extra_metrics = {}
        if feature_names is None:
            feature_names = []
        if output_names is None:
            output_names = []
        if output_units is None:
            output_units = {}
        lines = []
        lines.append(f"# Neural Network Execution Report")
        lines.append(f"- generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- seed: {self.config.get('seed')}")
        lines.append("")
        lines.append("## Purpose (in this app)")
        lines.append("- This neural network is a fast surrogate for the physics metrics.")
        lines.append("- Genetic optimization uses it to score many scaffold designs quickly, then verifies the best design with the physics simulation.")
        lines.append("")
        lines.append("## Model I/O (what it learns)")
        if feature_names:
            lines.append(f"- inputs ({len(feature_names)} scaffold/media parameters):")
            for name in feature_names:
                lines.append(f"  - {name}")
        else:
            lines.append(f"- inputs: {self.config['input_dim']} parameters (see app sidebar)")
        if output_names:
            lines.append(f"- outputs ({len(output_names)} predicted metrics):")
            for name in output_names:
                unit = output_units.get(name, "")
                if unit:
                    lines.append(f"  - {name} ({unit})")
                else:
                    lines.append(f"  - {name}")
        else:
            lines.append(f"- outputs: {self.config['output_dim']} metrics")
        lines.append("")
        lines.append("## Architecture")
        lines.append(f"- input_dim: {self.config['input_dim']}")
        lines.append(f"- hidden_layers: {self.config['hidden_layers']}")
        lines.append(f"- output_dim: {self.config['output_dim']}")
        lines.append(f"- activation: {self.config['activation']}")
        lines.append(f"- dropout: {self.config['dropout']}")
        lines.append(f"- attention: {self.config['use_attention']}")
        stack = [f"BatchNorm1d({self.config['input_dim']})"]
        if self.config["use_attention"]:
            stack.append(f"AttentionBlock({self.config['input_dim']})")
        prev = self.config["input_dim"]
        for width in self.config["hidden_layers"]:
            stack.append(f"Linear({prev}->{width})")
            stack.append(f"{self.config['activation']}")
            stack.append(f"Dropout(p={self.config['dropout']})")
            prev = width
        stack.append(f"Linear({prev}->{self.config['output_dim']})")
        lines.append(f"- layer_stack: {' | '.join(stack)}")
        lines.append("")
        lines.append("## Optimization")
        lines.append(f"- optimizer: {self.config['optimizer_name']}")
        lines.append(f"- learning_rate: {self.config['learning_rate']}")
        lines.append(f"- l2_weight_decay: {self.config['weight_decay']}")
        lines.append(f"- l1_lambda: {self.config['l1_lambda']}")
        lines.append(f"- scheduler: {self.config['scheduler_name']}")
        if self.config["scheduler_name"] != "none":
            lines.append(f"- scheduler_params: {self.config['scheduler_params']}")
        lines.append("")
        lines.append("## Performance")
        for k, v in {**self.last_train_metrics, **extra_metrics, **(extra_metrics or {})}.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Parameters")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lines.append(f"- total_params: {int(total_params)}")
        lines.append(f"- trainable_params: {int(trainable_params)}")
        lines.append("")
        for name, p in self.model.named_parameters():
            data = p.detach().cpu().numpy()
            lines.append(f"### {name}")
            lines.append(f"- shape: {tuple(data.shape)}")
            lines.append(f"- dtype: {data.dtype}")
            lines.append(f"- min: {float(np.min(data))}")
            lines.append(f"- max: {float(np.max(data))}")
            lines.append(f"- mean: {float(np.mean(data))}")
            lines.append(f"- std: {float(np.std(data))}")
            if include_weights:
                flat = data.reshape(-1)
                if flat.size <= max_tensor_elements:
                    lines.append("```")
                    lines.append(np.array2string(data, precision=6, separator=", ", max_line_width=140))
                    lines.append("```")
                else:
                    head = flat[: min(50, flat.size)]
                    tail = flat[-min(50, flat.size) :]
                    lines.append("```")
                    lines.append(f"[truncated] elements={flat.size} showing first/last {head.size}")
                    lines.append(np.array2string(head, precision=6, separator=", ", max_line_width=140))
                    lines.append("...")
                    lines.append(np.array2string(tail, precision=6, separator=", ", max_line_width=140))
                    lines.append("```")
            lines.append("")
        return "\n".join(lines)
    
    def save_model(self, path="slime_model.pth"):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path="slime_model.pth"):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False
