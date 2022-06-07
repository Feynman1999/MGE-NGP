from pathlib import Path
import os.path as osp
from mgenerf.utils.misc import is_str, is_list_of
from mgenerf.utils.path import mkdir_or_exist
from mgenerf.utils.dist import get_dist_info
from mgenerf.utils.time import get_time_str
from mgenerf.utils import LogBuffer
from .builder import obj_from_dict
from . import hooks
from .hooks import Hook, get_priority, CheckpointHook
from .load_save import load_checkpoint, save_checkpoint
from .utils.progressbar import ProgressBar
from megengine.distributed import group_barrier


class Trainer(object):
    def __init__(
        self,
        model,
        optimizer,
        grad_manager,
        work_dir,
        logger,
        lr_scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.grad_manager = grad_manager
        self.lr_scheduler = lr_scheduler

        if is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mkdir_or_exist(self.work_dir)
        else:
            raise TypeError("'work_dir' must be a str")

        self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()

        self.timestamp = get_time_str()
        
        self.logger = logger

        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        return self._model_name

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def hooks(self):
        return self._hooks

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def max_iters(self):
        return self._max_iters

    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority="NORMAL"):
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # Insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError(
                "'args' must be either a Hook object"
                " or dict, not {}".format(type(args))
            )

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, strict=False):
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.model, filename, strict)

    def save_checkpoint(self, out_dir, filename_tmpl="epoch_{}.pth", save_optimizer=True, meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)

    def train(self, data_loader, now_epoch, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        self._max_iters = self._max_epochs * self.length
        self.call_hook("before_train_epoch")

        base_step = now_epoch * self.length

        for i, data_batch in enumerate(data_loader):
            global_step = base_step + i
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(global_step)

            self._inner_iter = i

            self.call_hook("before_train_iter")

            outputs = self.model.train_step(data_batch, now_epoch = now_epoch, 
            gm = self.grad_manager, optim = self.optimizer, **kwargs)

            if not isinstance(outputs, dict):
                raise TypeError("model's train_step must return a dict")
            
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["batchsize"])
            
            self.outputs = outputs
            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")

        self.logger.info(f"work dir: {self.work_dir}")

        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader.dataset))

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook("before_val_iter")

            outputs = self.model.test_step(data_batch, **kwargs)

            for output in outputs:
                if self.rank == 0:
                    for _ in range(self.world_size):
                        prog_bar.update()

        group_barrier() # 同步

        # all_predictions = all_gather(detections)

        if self.rank != 0:
            return

        # save and eval

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        # torch.save(predictions, "final_predictions_debug.pkl")
        # TODO fix evaluation module
        output_dir = Path(self.work_dir) / "results" / f"epoch_{self.epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dict, _ = self.data_loader.dataset.evaluation(
            predictions, output_dir=output_dir
        )

        self.logger.info("\n")
        for k, v in result_dict["results"].items():
            self.logger.info(f"Evaluation {k}: {v}")

        self.call_hook("after_val_epoch")

    def resume(self, checkpoint, resume_optimizer=True):
        checkpoint = self.load_checkpoint(checkpoint)

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        assert isinstance(data_loaders, list)
        assert is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info("Start running, work_dir: %s", work_dir)
        self.logger.info("workflow: %s, max: %d epochs, now: %d epochs", workflow, max_epochs, self.epoch)
        self.call_hook("before_run")

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError("Trainer has no method named '{}' to run an epoch".format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError(
                        "mode in workflow must be a str or "
                        "callable function not '{}'".format(type(mode))
                    )

                for _ in range(epochs):
                    if mode == "train":
                        epoch_runner(data_loaders[i], self.epoch, **kwargs)
                    elif mode == "val":
                        epoch_runner(data_loaders[i], **kwargs) 
                    else:
                        raise NotImplementedError("")

        self.call_hook("after_run")

    def register_logger_hooks(self, log_config):
        log_interval = log_config["interval"]
        for info in log_config["hooks"]:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval)
            )
            self.register_hook(logger_hook, priority="VERY_LOW")

    def register_training_hooks(self, checkpoint_config=None, log_config=None):
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        if log_config is not None:
            self.register_logger_hooks(log_config)
