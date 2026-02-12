"""
Argument definitions for 3DGS training and rendering
"""

from argparse import ArgumentParser, Namespace
import os
import sys


class ParamGroup:
    """Base class for parameter groups"""

    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        """
        Initialize parameter group

        Args:
            parser: ArgumentParser instance
            name: Name of the group
            fill_none: Fill unspecified values with None
        """
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            arg_value = None if fill_none else value

            if isinstance(value, bool):
                if shorthand:
                    group.add_argument(
                        f"--{key}",
                        f"-{key[0:1]}",
                        default=arg_value,
                        action="store_true",
                    )
                else:
                    group.add_argument(
                        f"--{key}", default=arg_value, action="store_true"
                    )
            else:
                if shorthand:
                    group.add_argument(
                        f"--{key}", f"-{key[0:1]}", default=arg_value, type=type(value)
                    )
                else:
                    group.add_argument(f"--{key}", default=arg_value, type=type(value))

    def extract(self, args):
        """Extract this group's parameters from args"""
        group = Namespace()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    """Model parameters"""

    def __init__(self, parser: ArgumentParser, sentinel=False):
        self.sh_degree = 3  # Spherical harmonics degree (0-3)
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1  # -1 for original resolution
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters")


class PipelineParams(ParamGroup):
    """Pipeline parameters"""

    def __init__(self, parser: ArgumentParser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    """Optimization parameters"""

    def __init__(self, parser: ArgumentParser):
        self.iterations = 30_000  # Total training iterations
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lambda_dssim = 0.2  # SSIM loss weight

        # Densification parameters
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.opacity_cull = 0.005
        self.percent_dense = 0.01

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    """
    Get combined args from command line and cfg_args file

    Args:
        parser: ArgumentParser instance

    Returns:
        Namespace with combined args
    """
    cmdline_string = sys.argv[1:]
    cfgfilepath = ""
    args_cmdline = parser.parse_args(cmdline_string)

    # Try to find cfg_args file
    if args_cmdline.model_path:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        if os.path.exists(cfgfilepath):
            print(f"Config file found: {cfgfilepath}")
            with open(cfgfilepath) as cfg_file:
                print(f"Loading config: {cfgfilepath}")
                cfgfile_string = cfg_file.read()
                cfgfile_args = eval(cfgfile_string)

                # Merge args
                merged_dict = vars(cfgfile_args).copy()
                for key, value in vars(args_cmdline).items():
                    if value is not None:
                        merged_dict[key] = value
                return Namespace(**merged_dict)

    return args_cmdline
