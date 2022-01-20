import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from argparse import ArgumentParser
from face_recognition.model_interface import LitModel
from face_recognition.data_interface import *
from face_recognition.utils.utils import *
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import wandb


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Init data pipeline
    dm = LitDataModule(hparams=args)

    # Init LitModel
    model = LitModel(num_classes=dm.num_classes, hparams=args)
    print(model)

    checkpoint_callback = ModelCheckpoint(
        monitor='mean_val_acc',
        dirpath=append_path_by_date(args.model_ckpt_dir),
        filename='model-{epoch}-{mean_val_acc:.2f}',
        save_top_k=-1 if args.every_n_epochs==1 else 1,
        save_last=True,
        mode='max'
    )

    # Initialize wandb logger
    project_name = 'GP_AMSoft_TVConv' + args.dataset_name
    wandb_logger = WandbLogger(project=project_name, job_type='train')
    wandb_logger.log_hyperparams(args)

    # Initialize a trainer
    trainer = pl.Trainer(
        # fast_dev_run=1,
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=20,
        gpus=args.gpus,
        accelerator=args.accelerator,
        callbacks=[
            checkpoint_callback
        ],
        benchmark=True
    )

    trainer.fit(model, dm)

    # Close wandb run
    if args.test_phase == 0:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # NORMAL args
    parser.add_argument("--data_dir", type=str, default="./data")
    # parser.add_argument("--data_dir", type=str, default="./data/all_align_96")
    parser.add_argument("--input_size", type=list, default=[96, 96])
    parser.add_argument('-d', "--dataset_name", type=str, default='casia_webface_frontal',
                        choices=['casia_webface_frontal'])
    parser.add_argument("--val_data_dir_list", type=list,
                        default=['agedb_30', 'calfw', 'cfp_ff', 'lfw'])
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--drop_last", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gpus", type=int, default=-1,
                        help='how many gpu used among the visible gpus')
    parser.add_argument("--accelerator", type=str, default='ddp')
    parser.add_argument('-r', "--runs", type=int, default=1)
    parser.add_argument('-n', "--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1, choices=[0, 1])
    parser.add_argument("--every_n_epochs", type=int, default=0, choices=[0, 1])

    # frequently reset args
    parser.add_argument('-b', "--batch_size", type=int, default=512)
    parser.add_argument('-m', "--model_name", type=str, default="mobilenet_v2_x0_1")
    parser.add_argument('-g', "--gpu_id", type=str, default='1', help='visible gpu id')
    parser.add_argument('-a', "--atom", type=str, default='TVConv',
                        choices=['TVConv', 'base'])
    parser.add_argument("--drop_ratio", type=float, default=0)

    # TVConv related
    parser.add_argument("--TVConv_posi_chans", type=int, default=4, help='affinity maps')
    parser.add_argument("--TVConv_inter_chans", type=int, default=64)
    parser.add_argument("--TVConv_inter_layers", type=int, default=3)
    parser.add_argument("--TVConv_Bias", type=int, default=0, choices=[0, 1])

    parser.add_argument('--max_epochs', type=int, default=38)
    parser.add_argument('--lr_milestones', type=list, default=[22, 30, 35])

    args = parser.parse_args()

    for i in range(0, args.runs):
        print(f'Running progress: {i}/{args.runs}')
        train(args)