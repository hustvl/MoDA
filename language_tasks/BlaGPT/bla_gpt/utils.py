import ast
import importlib.util
import logging
import os
import sys
import warnings

import torch

warnings.filterwarnings(
    "ignore", category=SyntaxWarning, message="invalid escape sequence"
)

logger = logging.getLogger("base")


def detach_loss(x):
    """Detach and convert it to a float value if necessary."""
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return x.detach().item()
    return x


class RegisteredModelNameError(Exception):
    def __init__(self, name_error):
        super().__init__(
            f"Registered modules must start with `register_`. Incorrect registration: {name_error}"
        )


def register_model(func):
    if func.__name__.startswith("register_"):
        func._registered_model_name = func.__name__[9:]
        assert func._registered_model_name
    else:
        raise RegisteredModelNameError(func.__name__)
    func._registered_model = True
    return func


def find_project_root(current_file, marker_file=".git"):
    """Find the project root by searching for a marker file."""
    current_dir = os.path.dirname(os.path.abspath(current_file))
    while True:
        if os.path.exists(os.path.join(current_dir, marker_file)):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(
                f"Could not find project root containing {marker_file}"
            )
        current_dir = parent_dir


def find_registered_model_fns():
    """Find all functions with the register_model decorator in the project."""
    current_file = os.path.abspath(__file__)
    project_folder = find_project_root(current_file)
    registered_models = {}

    def process_file(file_path):
        try:
            with open(file_path, "r") as file:
                tree = ast.parse(file.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "register_model"
                        ):
                            model_name = node.name.replace("register_", "")
                            module_name = os.path.splitext(os.path.basename(file_path))[
                                0
                            ]

                            try:
                                spec = importlib.util.spec_from_file_location(
                                    module_name, file_path
                                )
                                module = importlib.util.module_from_spec(spec)
                                sys.modules[module_name] = module
                                spec.loader.exec_module(module)
                                register_function = getattr(module, node.name)
                                registered_models[model_name] = register_function
                            except ImportError as e:
                                logger.info(
                                    f"Error processing file {file_path}: {str(e)}"
                                )

        except Exception as e:
            logger.info(f"Error processing file {file_path}: {str(e)}")

    for root, _, files in os.walk(project_folder):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                process_file(file_path)

    return registered_models


def get_model(model_name):
    registered_fns = find_registered_model_fns()

    try:
        return registered_fns[model_name]()  # model, config
    except KeyError:
        raise ValueError(f"Model {model_name} not found in registered models. Registered models are: {list(registered_fns.keys())}")
