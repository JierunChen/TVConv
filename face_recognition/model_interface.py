from torch.optim.lr_scheduler import *
from face_recognition.model import *
from face_recognition.utils.utils import *
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional.classification.accuracy import accuracy


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model = globals()[self.hparams.model_name](input_size=self.hparams.input_size,
                                                        atom=self.hparams.atom,
                                                        TVConv_posi_chans=self.hparams.TVConv_posi_chans,
                                                        TVConv_inter_chans=self.hparams.TVConv_inter_chans,
                                                        TVConv_inter_layers=self.hparams.TVConv_inter_layers,
                                                        TVConv_Bias=bool(self.hparams.TVConv_Bias),
                                                        drop_ratio=self.hparams.drop_ratio
                                                        )
        self.head = AM_Softmax(classnum=num_classes, s=32.)
        self.criterion = nn.CrossEntropyLoss()

    # will be used during inference
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        out = self.head(out, y)
        loss = self.criterion(out, y)

        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)

        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc.detach(), on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        x1, x2, fold, flag = batch
        feature1 = self(x1)
        feature2 = self(x2)

        return feature1, feature2, fold, flag

    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs = self.all_gather(validation_step_outputs)
        # if self.trainer.is_global_zero and not self.trainer.sanity_checking:
        if not self.trainer.sanity_checking:
            acc_list = []
            for i, features in enumerate(validation_step_outputs):
                feature1s = []
                feature2s = []
                folds = []
                flags = []
                for batch in features:
                    feature1s.append(batch[0].view(-1, 512))
                    feature2s.append(batch[1].view(-1, 512))
                    folds.append(batch[2].view(-1))
                    flags.append(batch[3].view(-1))
                feature1s = torch.cat(feature1s, dim=0)
                feature2s = torch.cat(feature2s, dim=0)
                folds = torch.cat(folds, dim=0).detach().cpu().numpy()
                flags = torch.cat(flags, dim=0).detach().cpu().numpy()

                feature1s = l2_norm(feature1s.detach()).cpu().numpy()
                feature2s = l2_norm(feature2s.detach()).cpu().numpy()

                acc, mean_threshold = evaluation_10_fold(feature1s, feature2s, folds, flags,
                                                         method='l2_distance')

                acc_list.append(acc)
                self.log('val_acc_' + self.hparams.val_data_dir_list[i], acc, prog_bar=True)
                self.log('val_mean_threshold_' + self.hparams.val_data_dir_list[i], mean_threshold, prog_bar=True)

            self.log('mean_val_acc', sum(acc_list)/len(acc_list), prog_bar=True)

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
            nesterov=False
        )

        scheduler = MultiStepLR(optimizer, milestones=self.hparams.lr_milestones, gamma=0.1)

        return [optimizer], [scheduler]
