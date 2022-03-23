import logging
import os
import sys
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
from probing_project.constants import TASK_CLASSES
from probing_project.data.modules import ProbingModuleBase, get_module_class
from probing_project.model import ProbingModel
from probing_project.probes import get_probe_class
from probing_project.tasks import TaskBase, get_task_class
from probing_project.utils import check_volta_layer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from rich.logging import RichHandler


def parse_args():
    parser = ArgumentParser(
        description="Multimodal-Probes: Using several probes for measuring "
        "structural qualities in Multimodal BERT models.",
        add_help=False,
    )
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    # change some pytorch-lightning default values
    parser.set_defaults(
        max_epochs=40,
        gpus=-1,
        deterministic=True,
        benchmark=False,
        num_sanity_val_steps=0,
    )
    # add PROGRAM level args
    main_parser = parser.add_argument_group("MainArgs")
    # first we only do stuff for task, so we can add the correct help message
    main_parser.add_argument(
        "--task",
        required=True,
        type=str,
        choices=TASK_CLASSES,
        help="The task for which to train a probe.",
    )
    argv = [y for x in sys.argv for y in x.split("=")]
    if "--task" in argv:
        idx = argv.index("--task")
        args, _ = parser.parse_known_args(argv[idx : idx + 2])
        tmp_task = get_task_class(args.task)
        parser = tmp_task.add_task_specific_args(parser)
        del tmp_task
    else:
        parser = TaskBase.add_task_specific_args(parser)

    # After checking task, the probe argument should exist.
    # Check for it and see if we have to add probe specific argument
    if "--probe" in argv:
        idx = argv.index("--probe")
        # add a task argument to capture the missing required argument error
        args, _ = parser.parse_known_args(argv[idx : idx + 2] + ["--task", "TaskBase"])
        tmp_probe = get_probe_class(args.probe)
        parser = tmp_probe.add_probe_specific_args(parser)
        del tmp_probe

    # Add all the remaining Main Arguments, without worrying about faulty checks
    main_parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="call this flag to run in DEBUG mode. Sets logging level to debug, "
        "sets num-workers to 0, and prints extra statistics.",
    )
    main_parser.add_argument(
        "--JUST_PREPARE",
        action="store_true",
        help="Only prepare data for the current settings.",
    )
    main_parser.add_argument(
        "--seed", type=int, default=42, help="The random seed to use for everything."
    )
    main_parser.add_argument(
        "--use_wandb_logger",
        action="store_true",
        help="call this flag to stop logging to WandB.",
    )
    main_parser.add_argument(
        "--stopping_patience",
        type=int,
        default=5,
        help="Early Stopping patience. Terminates training after this many epochs "
        "without improvement. Uses stopping_monitor to check improvement.",
    )
    main_parser.add_argument(
        "--stopping_monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "spearmanr_mean_5-50", "root_acc", "uuas"],
        help="The metric used for keeping track of early stopping.",
    )
    main_parser.add_argument(
        "--stopping_delta",
        type=float,
        default=0.0001,
        help="minimum delta to count as improvement. Hewitt et al. default is 0.0001",
    )
    main_parser.add_argument(
        "--disable_val_bar",
        action="store_true",
        help="Stop showing sub progress bar for validation. "
        "Second progress bar causes issues in PyCharm.",
    )
    main_parser.add_argument(
        "--run_test",
        action="store_true",
        help="When called, the test split is also run.",
    )
    main_parser.add_argument(
        "--save_dir",
        type=str,
        default="results/",
        help="The directory where to store all the results and outputs.",
    )
    # We also add the help argument again, since we won't do any temporary parsing
    parser.add_argument("-h", "--help", action="help", help="Print this help message")

    # add DataModule/model specific args
    parser = ProbingModuleBase.add_datamodule_specific_args(parser)
    parser = ProbingModel.add_model_specific_args(parser)

    # Everything is added. Now we can parse all the arguments
    args = parser.parse_args()
    # This is important for checking if values are correct and
    # setting model_hidden_dim for the output of embedding models
    ProbingModuleBase.check_datamodule_args(args)

    # check if we have to do a debug run
    if args.DEBUG:
        # in debug mode, run the pytorch-lightning profiler
        parser.set_defaults(profiler="simple")

    # Create grouped arguments to easily pass the needed information to functions
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = dict()
        if group._group_actions:
            action_keys = [a.dest for a in group._group_actions]
            for k, v in vars(args).items():
                if k in action_keys:
                    group_dict[k] = v
        arg_groups[group.title] = group_dict
    # rich.print(arg_groups)
    return args, arg_groups


def init_logger(debug=False):
    if logging.getLogger("lightning").handlers:
        logging.getLogger("lightning").handlers.clear()
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler = RichHandler(rich_tracebacks=True)
    root_logger.addHandler(console_handler)
    log_handler = logging.getLogger(__name__)
    return log_handler


def get_run_name(args: Namespace, module: ProbingModuleBase):
    name = ""
    # first set the debug flag
    if args.DEBUG:
        name += "DEBUG_"
    # set the dataset used
    name += f"{args.dataset}_"
    # set bert model used
    name += f"{module.model_name}_"
    # set embeddings model used
    name += f"{args.embeddings_model}_"
    # set embeddings layer number used
    name += f"{args.bert_model_layer}_"
    # set embeddings process model used
    name += f"{args.embedding_process_model}_"
    # set task info
    name += f"{args.task}_"
    # set probe used
    name += f"{args.probe}"
    if "probe_rank" in args:
        name += f"-{args.probe_rank}"
    if "probe_hidden_layers" in args:
        name += f"-{args.probe_hidden_layers}"
    if "probe_dropout" in args:
        name += f"-{args.probe_dropout}"
    # set some general info
    name += (
        f"_bs{args.batch_size}_ep{args.max_epochs}_stop{args.stopping_monitor}"
        f"_pa{args.stopping_patience}_del{args.stopping_delta}_seed{args.seed}"
    )
    return name


def main(args, group_args):
    # make sure everything is as deterministic as possible
    _ = pl.seed_everything(args.seed)

    task_class = get_task_class(args.task)
    task = task_class()

    # create the datamodule
    module_class = get_module_class(args.dataset)
    datamodule = module_class(task=task, **group_args["ModuleArgs"])

    if args.JUST_PREPARE:
        datamodule.prepare_data()
        return 0

    # create the logger
    if args.use_wandb_logger:
        # We used WandB Logger in our project, update these settings
        # if you want to use it as well. Some settings are set in environment variables
        run_name = get_run_name(args, datamodule)
        torch_logger = WandbLogger(
            name=run_name,
            save_dir=args.save_dir,
            project="multimodal-probes",
            entity="liir-kuleuven",
            offline=args.DEBUG,
        )
        # change the directory where logger saves file for current run
        log_dir = os.path.join(
            torch_logger.save_dir,
            torch_logger.experiment.name,
            torch_logger.experiment.dir.split("/")[-2],
        )
        torch_logger.experiment.config.log_dir = log_dir
        args.results_root_dir = log_dir
        os.makedirs(log_dir)
    else:
        log_dir = args.save_dir



    # create the model
    model = ProbingModel(
        task=task,
        results_root_dir=log_dir,
        embedding_process_model=args.embedding_process_model,
        probe=args.probe,
        probe_group_args=group_args["ProbeArgs"],
        model_hidden_dim=args.model_hidden_dim,
        use_only_caption_regions=args.use_only_caption_regions,
    )

    if not args.no_wandb_logger:
        torch_logger.watch(model)

    # CHECK IF WE SHOULD RUN THIS RUN, OR JUST SKIP IT
    try:
        check_volta_layer(
            config=datamodule.mm_config,
            task=args.task,
            layer_idx=args.bert_model_layer,
            mm_bert_layer=datamodule.mm_map_bert_layer,
        )
    except ValueError:
        sys.exit(1)

    # Create the needed pytorch-lightning callbacks
    check = ModelCheckpoint(
        dirpath=log_dir,
        filename="{{epoch:02d}}-{{{}:.2f}}".format(args.stopping_monitor),
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor=args.stopping_monitor,
        mode="min" if "loss" in args.stopping_monitor else "max",
    )
    es = EarlyStopping(
        monitor=args.stopping_monitor,
        patience=args.stopping_patience,
        min_delta=args.stopping_delta,
        verbose=True,
        mode="min" if "loss" in args.stopping_monitor else "max",
    )

    # Create the trainer
    trainer = pl.Trainer.from_argparse_args(
        args=args, logger=torch_logger, callbacks=[RichProgressBar(), es, check]
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)
    logger.info("Finished Training!")

    # Start Testing
    if args.run_test:
        logger.info("Starting run on test set.")
        trainer.test(model, datamodule=datamodule)
        logger.info("Testing finished. Program Complete!")
    else:
        logger.info('Not testing, set "--run_test" to run test. Program Complete!')


if __name__ == "__main__":
    parsed_args, parsed_group_args = parse_args()
    logger = init_logger(parsed_args.DEBUG)

    os.makedirs(parsed_args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exit(main(args=parsed_args, group_args=parsed_group_args))
