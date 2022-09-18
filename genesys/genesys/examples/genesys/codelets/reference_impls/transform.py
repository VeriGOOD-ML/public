
import numpy as np
from functools import partial
from . import ReferenceOp, quantize_np

class Transform(ReferenceOp):

    def __init__(self, transform_type, cdlt, program):
        self.transform_type = transform_type
        self.dtype = "FXP32"
        self.axis = self.cdlt.required_params['axis'].value
        operands = [cdlt.inputs[0]]
        outputs = [cdlt.outputs[0]]
        super().__init__(cdlt, operands, outputs, program)


    def fn_impl(self, inouts):
        data = inouts['inputs'][0].data
        out_shape = self.outputs[0].shape
        if len(data.shape) == 4:
            data = data.transpose((0, 3, 1, 2))

        if self.transform_type == "reshape":
            out = data.reshape(out_shape)
        elif self.transform_type == "squeeze":
            out = np.squeeze(data)
        elif self.transform_type == "where":
            x = inouts['inputs'][1].data
            cond = data
            out = np.where(cond, x)
        elif self.transform_type == "concat":
            out = data.copy()
        elif self.transform_type == "resize":
            out = data.copy()
        elif self.transform_type == 'split':
            out = data.copy()
        else:
            raise RuntimeError("unknown reduction type")

        if len(out.shape) == 4:
            out = out.transpose((0, 2, 3, 1))
        inouts['outputs'] = [out]
        return inouts

def load_transform_impls(cfg):

    TRANSFORM_IMPLS = {
        'tensor_reshape4d2d': partial(Transform, 'reshape'),
        'tensor_reshape4d3d': partial(Transform, 'reshape'),
        'tensor_reshape3d4d': partial(Transform, 'reshape'),
        'tensor_reshape3d2d': partial(Transform, 'reshape'),
        'tensor_reshape2d3d': partial(Transform, 'reshape'),
        'split': partial(Transform, 'split'),
        # 'tensor_flip': tensor_flip,
        'elem_where': partial(Transform, 'where'),
        # 'tensor_pad': tensor_pad,
        'concat': partial(Transform, 'concat'),
        'tensor_squeeze' : partial(Transform, 'squeeze'),
        'resize': partial(Transform, 'resize')
    }
    return TRANSFORM_IMPLS