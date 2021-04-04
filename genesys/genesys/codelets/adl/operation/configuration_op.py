from .base_op import Operation
from codelets.adl.flex_param import FlexParam


class Configure(Operation):

    def __init__(self, start_or_finish,
                 target=None,
                 add_codelet=True,
                 **kwargs
                 ):
        self._target_name = target
        self._start_or_finish = start_or_finish
        required_params = []
        resolved_params = {}
        for k, v in kwargs.items():
            if v is None:
                required_params.append(k)
            else:
                resolved_params[k] = FlexParam(k)
                resolved_params[k].value = v

        for k in list(kwargs.keys()):
            if k not in Operation.BASE_KWARG_NAMES:
                kwargs.pop(k)

        super(Configure, self).__init__('config', required_params,
                                        target=target,
                                        resolved_params=resolved_params,
                                        add_codelet=add_codelet, **kwargs)
    @property
    def target_name(self):
        return self._target_name

    @property
    def start_or_finish(self):
        return self._start_or_finish

    def get_config_param_value(self, param_name):
        if param_name not in self.resolved_params:
            raise RuntimeError(f"Unable to get value for {param_name} in config op. Possible params:\n"
                               f"{list(self.resolved_params.keys())}")
        return self.resolved_params[param_name].value

    def op_type_params(self):
        op_params = [f"{self.start_or_finish}"]
        return op_params

    def evaluate_parameters(self, node, hag, cdlt):
        pass

    def emit(self, output_type):
        # TODO: Add template
        if output_type == "operations":
            op_str = f"{self.op_str}: {self.start_or_finish}-{self.target}"
        elif output_type == "json":
            op_str = {"op_type": self.op_type,
                      "op_id": self.global_op_id,
                      "start_or_finish": self.start_or_finish,
                      "target": self.target}
        else:
            op_str = []
            for ft in self.instructions:
                ft_out = ft.emit(output_type)
                op_str += ft_out
        return op_str

    def copy(self, cdlt, target=None, start_or_finish=None, **kwargs):
        obj = super(Configure, self).copy(cdlt, **kwargs)
        obj._target_name = target or self.target
        obj._start_or_finish = start_or_finish or self.start_or_finish
        return obj

