from src.utils.misc import convert_normalized_tensor_to_plottable_array
from tqdm import tqdm
import sty
from sty import fg, rs, ef
from abc import ABC
import numpy as np
import pathlib
import torch
from time import time
import src.utils.misc as utils
import signal, os
import time
import math

try:
    import neptune
except:
    pass


class CallbackList(object):
    """Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
    """

    def __init__(self, callbacks):
        self.callbacks = [c for c in callbacks]

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_optimizer(self, model):
        for callback in self.callbacks:
            callback.set_optimizer(model)

    def set_loss_fn(self, model):
        for callback in self.callbacks:
            callback.set_loss_fn(model)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_training_step_end(self, batch, logs=None):
        """Called after training is finished, but before the batch is ended.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_training_step_end(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_training_step_end(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class StopFromUserInput(Callback):
    stop_next_iter = False

    def __init__(self):
        super().__init__()
        signal.signal(signal.SIGINT, self.handler)  # CTRL + C

    def handler(self, signum, frame):
        self.stop_next_iter = True

    def on_batch_end(self, batch, logs=None):
        if self.stop_next_iter:
            logs["stop"] = True
            print("Stopping from user input")
            # raise Exception


class TriggerActionWhenReachingValue(Callback):
    def __init__(
        self,
        value_to_reach,
        metric_name,
        mode="max",
        patience=1,
        check_after_batch=True,
        action=None,
        action_name="",
        check_every=1,
    ):
        self.patience = patience
        self.check_every = check_every
        self.action = action
        self.action_name = action_name
        self.count_patience = 0
        self.mode = mode
        self.value_to_reach = value_to_reach
        self.metric_name = metric_name
        self.check_after_batch = check_after_batch
        self.check_idx = 0
        print(
            fg.green
            + f"Action [{self.action_name}] when [{self.metric_name}] has reached value {'higher' if self.mode == 'max' else 'lower'} than [{self.value_to_reach}] for {self.patience} checks (checked every {self.check_every} {'batches' if self.check_after_batch else 'epoches'})"
            + rs.fg
        )
        super().__init__()

    def compare(self, metric, value):
        if self.mode == "max":
            return metric >= value
        if self.mode == "min":
            return metric <= value

    def check_and_stop(self, logs=None):
        self.check_idx += 1
        if self.check_idx >= self.check_every:
            self.check_idx = 0
            if self.compare(logs[self.metric_name], self.value_to_reach):
                self.count_patience += 1
                # print(f'PATIENCE +1 : {self.count_patience}/{self.patience}')
                if self.count_patience >= self.patience:
                    logs["stop"] = True
                    print(
                        fg.green
                        + f"\nMetric [{self.metric_name}] has reached value {'higher' if self.mode == 'max' else 'lower'} than [{self.value_to_reach}]. Action [{self.action_name}] triggered"
                        + rs.fg
                    )
            else:
                self.count_patience = 0

    def on_batch_end(self, batch, logs=None):
        if self.check_after_batch:
            self.check_and_stop(logs)

    def on_epoch_end(self, epoch, logs=None):
        if not self.check_after_batch:
            self.check_and_stop(logs)


class TriggerActionWithPatience(Callback):
    def __init__(
        self,
        mode="min",
        min_delta=0,
        patience=10,
        min_delta_is_percentage=False,
        metric_name="nept/mean_acc",
        check_every=100,
        triggered_action=None,
        action_name="",
        weblogger=False,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.triggered_action = triggered_action
        self.mode = mode  # mode refers to what are you trying to reach.
        self.check_every = check_every
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_iters = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, min_delta_is_percentage)
        self.metric_name = metric_name
        self.action_name = action_name
        self.first_iter = True
        self.weblogger = weblogger
        self.exp_metric = None
        self.patience = self.patience // self.check_every
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False
        self.string = f"Action {self.action_name} for metric [{self.metric_name}] <> {self.mode}, checking every [{self.check_every} batch iters], patience: {self.patience} [corresponding to [{patience}] batch iters]]"
        print(f"Set up action: {self.string}")

    def on_batch_end(self, batch, logs=None):
        if self.metric_name not in logs:
            return True

        if logs["tot_iter"] % self.check_every == 0:
            metrics = logs[self.metric_name].value
            (
                print(f"Iter: {logs['tot_iter']}, Metric: {logs[self.metric_name]}")
                if self.verbose
                else None
            )

            if self.best is None:
                self.best = metrics
                return

            if self.is_better(metrics, self.best):
                self.num_bad_iters = 0  # bad epochs: does not 'improve'
                self.best = metrics
            else:
                self.num_bad_iters += 1
            print(f"Num Bad Iter: {self.num_bad_iters}") if self.verbose else None
            (
                print(f"Patience: {self.num_bad_iters}/{self.patience}")
                if (self.verbose or self.patience - self.num_bad_iters < 20)
                else None
            )

            if self.num_bad_iters >= self.patience:
                print(f"Action triggered: {self.string}")
                self.triggered_action(logs, self)
                # needs to reset itself
                self.num_bad_iters = 0
        else:
            (
                print(
                    f"Not updating now {self.check_every - (logs['tot_iter'] % self.check_every)}"
                )
                if self.verbose
                else None
            )

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


from torch.optim.lr_scheduler import ReduceLROnPlateau


class PlateauLossLrScheduler(Callback):
    def __init__(self, optimizer, check_batch=False, patience=2, loss_metric="loss"):
        self.loss_metric = loss_metric
        self.scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        self.last_lr = [i["lr"] for i in self.scheduler.optimizer.param_groups]
        self.check_batch = check_batch

    def on_epoch_end(self, epoch, logs=None):
        if not self.check_batch:
            self.check_and_update(logs)

    def check_and_update(self, logs):
        self.scheduler.step(logs[self.loss_metric])
        if self.last_lr != [i["lr"] for i in self.scheduler.optimizer.param_groups]:
            print(
                (fg.blue + "learning rate: {} => {}" + rs.fg).format(
                    self.last_lr,
                    [i["lr"] for i in self.scheduler.optimizer.param_groups],
                )
            )
            self.last_lr = [i["lr"] for i in self.scheduler.optimizer.param_groups]

    def on_batch_end(self, batch, logs):
        if self.check_batch:
            self.check_and_update(logs)


class StopWhenMetricIs(Callback):
    def __init__(self, value_to_reach, metric_name, check_after_batch=True):
        self.value_to_reach = value_to_reach
        self.metric_name = metric_name
        self.check_after_batch = check_after_batch
        print(
            fg.cyan
            + f"This session will stop when metric [{self.metric_name}] has reached the value  [{self.value_to_reach}]"
            + rs.fg
        )
        super().__init__()

    def check_and_stop(self, logs=None):
        if logs[self.metric_name] >= self.value_to_reach:
            logs["stop"] = True
            print(
                f"Metric [{self.metric_name}] has reached the value [{self.value_to_reach}]. Stopping"
            )

    def on_batch_end(self, batch, logs=None):
        if self.check_after_batch:
            self.check_and_stop(logs)

    def on_epoch_end(self, epoch, logs=None):
        if not self.check_after_batch:
            self.check_and_stop(logs)


class SaveModelAndOpt(Callback):
    def __init__(
        self,
        net,
        output_folder,
        loss_metric_name="loss",
        print_save=False,
        optimizers=None,  # a list of Optimizers
        epsilon_loss=0.1,
        min_iter=np.inf,
        max_iter=np.inf,
        at_epoch_end=True,
    ):
        self.print_save = print_save
        self.epoch_end = at_epoch_end
        self.output_folder = output_folder
        self.net = net
        self.last_loss = np.inf
        self.last_iter = 0
        self.min_iter = min_iter
        self.epsilone_loss = epsilon_loss
        self.optimizers = optimizers
        self.loss_metric_name = loss_metric_name
        self.max_iter = max_iter
        super().__init__()

    def save_model(self, path: pathlib.Path, print_it=True):
        # pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        (
            print(
                fg.yellow
                + ef.inverse
                + "Saving MODEL in {}".format(path)
                + rs.fg
                + rs.inverse
            )
            if print_it
            else None
        )
        torch.save(
            self.net.state_dict(),
            path,
        )

    def save_optimizers(self, path: pathlib.Path, print_it=True):
        # pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        (
            print(
                fg.yellow
                + ef.inverse
                + "Saving ALL OPTIMIZERS in {}".format(path)
                + rs.fg
                + rs.inverse
            )
            if print_it
            else None
        )
        torch.save(
            [opt.state_dict() for opt in self.optimizers],
            path,
        )

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_end:
            self.save_model(pathlib.Path(self.output_folder) / "checkpoint_model.pt")
            if self.optimizers:
                self.save_optimizers(
                    pathlib.Path(self.output_folder) / "checkpoint_optimizers.pt"
                )

    def on_batch_end(self, batch, logs=None):
        if self.output_folder is not None:
            if (
                ((logs["tot_iter"] - self.last_iter) > self.max_iter)
                or ((self.last_loss - logs[self.loss_metric_name]) > self.epsilone_loss)
                and ((logs["tot_iter"] - self.last_iter) > self.min_iter)
            ):
                self.last_iter = logs["tot_iter"]
                self.last_loss = logs[
                    self.loss_metric_name
                ]  # .value  ## ouch! You cannot overload assignment operator :( ! Nope!
                self.save_model(
                    pathlib.Path(self.output_folder) / "checkpoint_model.pt",
                    print_it=self.print_save,
                )
                self.save_optimizers(
                    pathlib.Path(self.output_folder) / "checkpoint_optimizers.pt",
                    print_it=self.print_save,
                )

    def on_train_end(self, logs=None):
        if self.output_folder is not None:
            self.save_model(
                str(pathlib.Path(self.output_folder) / "checkpoint_model.pt")
            )
            self.save_model(str(pathlib.Path(self.output_folder) / "finished_model.pt"))
            self.save_optimizers(
                str(pathlib.Path(self.output_folder) / "checkpoint_optimizers.pt")
            )
            self.save_optimizers(
                str(pathlib.Path(self.output_folder) / "finished_optimizers.pt")
            )


class PrintLogs(Callback, ABC):
    def __init__(self, id, plot_every=100, plot_at_end=True):
        self.id = id
        self.last_iter = 0
        self.plot_every = plot_every
        self.plot_at_end = plot_at_end

    def on_training_step_end(self, batch, logs=None):
        if logs["tot_iter"] - self.last_iter > self.plot_every:
            # self.print_logs(self.get_value(self.running_logs[self.id]), logs)
            self.print_logs(logs[self.id], logs)
            # self.running_logs[self.id] = []
            self.last_iter = logs["tot_iter"]

    def on_train_end(self, logs=None):
        if self.plot_at_end:
            self.print_logs(logs[self.id], logs)


class PrintConsole(PrintLogs):
    def __init__(self, endln="\n", **kwargs):
        self.endln = endln
        super().__init__(**kwargs)

    def print_logs(self, values, logs):
        if isinstance(values, str):
            value_format = values
        elif isinstance(values, int):
            value_format = f"{values}"
        else:
            value_format = f"{values:.3}"
        print(fg.cyan + f"{self.id}: {value_format}" + rs.fg, end=self.endln)


class ClipGradNorm(Callback):
    def __init__(self, net, max_norm):
        self.net = net
        self.max_norm = max_norm

    def on_training_step_end(self, batch, logs=None):
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)


class ProgressBar(Callback):
    def __init__(self, l, batch_size, logs_keys=None):
        if l is None:
            tt = None
        else:
            tt = l
        self.pbar = tqdm(total=tt, dynamic_ncols=True)
        self.batch_size = batch_size
        self.logs_keys = logs_keys if logs_keys is not None else []
        # self.length_bar = l
        # self.pbar.bar_format = "{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_inv_fmt}{postfix}]"

    def on_training_step_end(self, batch_index, batch_logs=None):
        # framework_utils.progress_bar(batch_index, self.length_bar)
        self.pbar.set_postfix_str(
            " / ".join(
                [
                    sty.fg.cyan + f"{lk}:{batch_logs[lk]:.5f}" + sty.rs.fg
                    for lk in self.logs_keys
                ]
            )
        )
        self.pbar.set_description(
            sty.fg.red + f'Epoch {batch_logs["epoch"]}' + sty.rs.fg
        )
        self.pbar.update(self.batch_size)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.reset()

    def on_train_end(self, logs=None):
        self.pbar.close()


from csv import writer


class SaveInfoCsv(Callback):
    def __init__(
        self,
        log_names,
        path,
        update_on_batch_end=False,
        update_on_epoch_end=True,
        update_on_train_end=False,
    ):
        self.log_names = log_names
        self.path = path
        self.rows = []
        self.update_on_batch_end = update_on_batch_end
        self.update_on_train_end = update_on_train_end
        self.update_on_epoch_end = update_on_epoch_end
        # pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

        f = open(self.path, "w")
        csv_writer = writer(f)
        csv_writer.writerow(log_names)

    def append_list_as_row(self, file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, "a+", newline="") as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)

    def update(self, logs):
        self.append_list_as_row(self.path, [logs[l] for l in self.log_names])

    def on_epoch_end(self, epoch, logs=None):
        if self.update_on_epoch_end:
            self.update(logs)

    def on_batch_end(self, batch, logs=None):
        if self.update_on_batch_end:
            self.update(logs)

    def on_train_end(self, logs=None):
        if self.update_on_train_end:
            self.update(logs)


class PrintNeptune(PrintLogs):
    def __init__(self, weblogger, convert_str=False, log_prefix="", **kwargs):
        self.convert_str = convert_str
        self.weblogger = weblogger
        self.log_prefix = log_prefix
        super().__init__(**kwargs)

    def print_logs(self, values, logs):
        if isinstance(self.weblogger, neptune.Run):
            if self.convert_str:
                self.weblogger[self.log_prefix + self.id].log(str(values))
            else:
                self.weblogger[self.log_prefix + self.id].log(values)


class DuringTrainingTest(Callback):
    test_time = 0
    num_tests = 0

    def __init__(
        self,
        testing_loaders,
        every_x_epochs=None,
        eval_mode=True,
        every_x_iter=None,
        every_x_sec=None,
        weblogger=False,
        logs_prefix="rndtxt",
        multiple_sec_of_test_time=None,
        auto_increase=False,
        log_text="",
        call_run=None,
        callbacks=None,
        compute_conf_mat=True,
        plot_samples_corr_incorr=False,
        first_epoch_test=True,
    ):
        self.first_epoch_test = first_epoch_test
        self.eval_mode = eval_mode
        self.callbacks = [] if callbacks is None else callbacks
        self.testing_loader = testing_loaders
        self.compute_conf_mat = compute_conf_mat
        self.logs_prefix = logs_prefix
        self.every_x_epochs = every_x_epochs
        self.auto_increase = auto_increase  # this is also used as a multiplier.
        self.every_x_iter = every_x_iter
        self.every_x_sec = every_x_sec
        if self.auto_increase:
            self.every_x_sec = 20
        self.weblogger = weblogger
        self.log_text = log_text
        self.call_run = call_run
        self.time_from_last_test = None
        self.multiple_sec_of_test_time = multiple_sec_of_test_time
        self.plot_samples_corr_incorr = plot_samples_corr_incorr

    def on_train_begin(self, logs=None):
        self.time_from_last_test = time.time()

    def get_callbacks(self, log, testing_loader):
        cb = self.callbacks + [
            StopWhenMetricIs(
                value_to_reach=0,
                metric_name=f"{self.logs_prefix}epoch",
                check_after_batch=False,
            )
        ]
        # cb.append(PlotImagesEveryOnceInAWhile(self.weblogger, testing_loader.dataset, plot_every=1, plot_only_n_times=1, plot_at_the_end=False, max_images=20, text=f"Test no. {self.num_tests}")) if self.plot_samples_corr_incorr else None
        return cb

    def run_tests(self, logs, last_test=False):
        start_test_time = time.time()
        print(fg.green, end="")
        print(
            f"\n## Testing "
            + fg.green
            + f"[{self.testing_loader.dataset.name}]"
            + ((" in ~EVAL~ mode") if self.eval_mode else (" in ~TRAIN~ mode"))
            + " ##"
        )
        print(rs.fg, end="")

        def test(testing_loader, log="", last_test=False):
            mid_test_cb = self.get_callbacks(log, testing_loader)

            with torch.no_grad():
                _, logs_test = self.call_run(
                    testing_loader,
                    train=False,
                    callbacks=mid_test_cb,
                    logs=logs,
                    logs_prefix=self.logs_prefix,
                    collect_data=True if self.plot_samples_corr_incorr else False,
                )

        if self.eval_mode:
            self.model.eval()
            test(
                self.testing_loader,
                log=f" EVALmode [{self.testing_loader.dataset.name}]",
                last_test=last_test,
            )

        if not self.eval_mode:
            self.model.train()
            test(
                self.testing_loader,
                log=f" TRAINmode [{self.testing_loader.dataset.name}]",
                last_test=last_test,
            )

        self.model.train()
        self.num_tests += 1

        self.time_from_last_test = time.time()
        self.test_time = time.time() - start_test_time
        if self.auto_increase and "tot_iter" in logs:
            self.every_x_sec = self.auto_increase * (
                self.test_time
                + 0.5 * self.test_time * math.log(logs["tot_iter"] + 1, 1.2)
            )
            print(
                "Test time is {:.4f}s, next test is gonna happen in {:.4f}s".format(
                    self.test_time, self.every_x_sec
                )
            )

        if self.multiple_sec_of_test_time:
            print(
                "Test time is {:.4f}s, next test is gonna happen in {:.4f}s".format(
                    self.test_time, self.test_time * self.multiple_sec_of_test_time
                )
            )
        print(fg.green + "\n#############################################" + rs.fg)

    def on_epoch_begin(self, epoch, logs=None):
        if (self.every_x_epochs is not None and epoch % self.every_x_epochs == 0) or (
            epoch == 0 and self.first_epoch_test
        ):
            (
                print(f"\nTest every {self.every_x_epochs} epochs")
                if self.every_x_epochs is not None
                else None
            )
            self.run_tests(logs)

    def on_batch_end(self, batch, logs=None):
        if (
            (self.every_x_iter is not None and logs["tot_iter"] % self.every_x_iter)
            or (
                self.every_x_sec is not None
                and self.every_x_sec < time.time() - self.time_from_last_test
            )
            or (
                self.multiple_sec_of_test_time is not None
                and time.time() - self.time_from_last_test
                > self.multiple_sec_of_test_time * self.test_time
            )
        ):
            if self.every_x_iter is not None and logs["tot_iter"] % self.every_x_iter:
                print(f"\nTest every {self.every_x_iter} iterations")
            if (
                self.every_x_sec is not None
                and self.every_x_sec < time.time() - self.time_from_last_test
            ):
                print(
                    f"\nTest every {self.every_x_sec} seconds ({(time.time() -self.time_from_last_test):.3f} secs passed from last test)"
                )
            if (
                self.multiple_sec_of_test_time is not None
                and time.time() - self.time_from_last_test
                > self.multiple_sec_of_test_time * self.test_time
            ):
                print(
                    f"\nTest every {self.multiple_sec_of_test_time * self.test_time} seconds ({time.time() - self.time_from_last_test} secs passed from last test)"
                )

            self.run_tests(logs)

    def on_train_end(self, logs=None):
        self.run_tests(logs, last_test=True)
