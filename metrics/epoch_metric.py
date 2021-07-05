from typing import Callable, Sequence
from functools import wraps
import torch.distributed as dist
import torch
import pickle

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    rank = dist.get_rank()
    device = torch.device('cuda', rank)


    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    ret = []
    for d in data_list:
        ret.append(d.clone().to(device))
    return ret


class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """


def reinit__is_reduced(func: Callable) -> Callable:
    """Helper decorator for distributed configuration.
    See :doc:`metrics` on how to use it.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._is_reduced = False

    wrapper._decorated = True
    return wrapper


class EpochMetric:
    """Class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output
    datatype should be `float32`. Target datatype should be `long`.
    .. warning::
        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.
    .. warning::
        Current implementation does not work with distributed computations. Results are not gather across all devices
        and computed results are valid for a single device only.
    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`.
    Args:
        compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
            `predictions` and `targets` and returns a scalar.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
    """

    def __init__(self, compute_fn: Callable, output_transform: Callable = lambda x: x):

        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable.")

        self._is_reduced = False
        self.compute_fn = compute_fn
        self.reset()

    @reinit__is_reduced
    def reset(self) -> None:
        self._predictions = []
        self._targets = []

    def _check_shape(self, output):
        y_pred, y = output
        if y_pred.ndimension() not in (1, 2):
            raise ValueError("Predictions should be of shape (batch_size, n_classes) or (batch_size, ).")

        if y.ndimension() not in (1, 2):
            raise ValueError("Targets should be of shape (batch_size, n_classes) or (batch_size, ).")

        if y.ndimension() == 2:
            if not torch.equal(y ** 2, y):
                raise ValueError("Targets should be binary (0 or 1).")

    def _check_type(self, output):
        y_pred, y = output
        if len(self._predictions) < 1:
            return
        dtype_preds = self._predictions[-1].type()
        if dtype_preds != y_pred.type():
            raise ValueError(
                "Incoherent types between input y_pred and stored predictions: "
                "{} vs {}".format(dtype_preds, y_pred.type())
            )

        dtype_targets = self._targets[-1].type()
        if dtype_targets != y.type():
            raise ValueError(
                "Incoherent types between input y and stored targets: " "{} vs {}".format(dtype_targets, y.type())
            )

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        y_pred = y_pred.clone().to(y_pred.device)
        y = y.clone().to(y_pred.device)

        self._check_type((y_pred, y))
        self._predictions.append(y_pred)
        self._targets.append(y)

    def compute(self) -> Sequence[torch.Tensor]:

        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError("EpochMetric must have at least one example before it can be computed.")

        rank = dist.get_rank()
        device = torch.device('cuda', rank)

        _prediction_tensor = torch.cat(self._predictions, dim=0).to(device).view(-1)
        _target_tensor = torch.cat(self._targets, dim=0).to(device).view(-1)

        ws = dist.get_world_size()

        dist.barrier()
        if ws > 1 and not self._is_reduced:
            _prediction_output = all_gather(_prediction_tensor)
            _target_output = all_gather(_target_tensor)

            _prediction_tensor = torch.cat(_prediction_output, dim=0)
            _target_tensor = torch.cat(_target_output, dim=0)

        self._is_reduced = True
        _prediction_tensor = _prediction_tensor.cpu()
        _target_tensor = _target_tensor.cpu()

        result = torch.zeros(1).to(device)
        if dist.get_rank() == 0:
            # Run compute_fn on zero rank only
            result = self.compute_fn(_prediction_tensor, _target_tensor)

        result = torch.tensor(result.item()).to(device)
        if ws > 1:
            dist.broadcast(result, src=0)

        _prediction_tensor = _prediction_tensor.numpy()
        _target_tensor = _target_tensor.numpy()
        return result.item(), _prediction_tensor, _target_tensor

    @property
    def predictions(self):
        return self._predictions

    @property
    def targets(self):
        return self._targets
