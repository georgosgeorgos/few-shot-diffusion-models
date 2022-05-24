from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import wandb
import numpy as np

class Logger:
    def __init__(self, args, lst, cfg):
        self.args = args
        self.lst = lst
        self.cfg = cfg
        self.metrics={"epoch" : [],
                      "train": {l: [] for l in lst},
                      "val": {l: [] for l in lst},
                      "test": {l: [] for l in lst}
                    }
        self.best_vlb = -float('Inf')
        
        # self.den = {"train": (self.args.len_tr // self.args.batch_size),
        #             "val": (self.args.len_vl // self.args.batch_size),
        #             "test": (self.args.len_ts // self.args.batch_size)
        #             }
        # # if drop-last is false
        # for d in self.den:
        #     if self.den[d] == 0:
        #         self.den[d] = 1

    def init_writer(self):
        # set writer tensorboard for visualizations during training
        #self.writer_tb = SummaryWriter(self.args.run_dir)
        # set weight and biases
        wb_name = self.args.name + "_" + self.args.timestamp
        if self.args.tag != "": 
            wb_name = wb_name + "_" + self.args.tag 
        self.writer_wb = wandb.init(dir=self.args.run_dir, name=wb_name, config=self.cfg)

    def log_grad(self, model):
        self.writer_wb.watch(model, log='all')
    
    def close_writer(self):
        #self.writer_tb.close()
        self.writer_wb.finish()
        
    def add_log(self, log, split, epoch):
        for l in self.lst:
            # log 
            k = l + "/" + split

            log[l] = np.mean(log[l])
            self.metrics[split][l].append(log[l])
            
            #self.writer_tb.add_scalar(k, log[l], epoch)
            self.writer_wb.log({k: log[l]}, step=epoch)

        if split == "train":
            self.metrics["epoch"].append(epoch)

    # dict are passed by reference
    def add_logs(self, train_log, val_log, test_log, epoch):
        self.add_log(train_log, "train", epoch)
        self.add_log(val_log, "val", epoch)
        self.add_log(test_log, "test", epoch)

    def update_best(self, test_log):
        # maximize the lower-bound
        if test_log["vlb"] > self.best_vlb:
            self.best_vlb = test_log["vlb"]
            for l in self.lst:
                self.writer_wb.summary[l] = test_log[l]

    def get_metric(self):
        return self.metrics

    # def add_sample(self, samples, samples_mnist, n=10):
    #     # log samples
    #     epoch = self.metrics["epoch"][-1]
        
    #     grid = make_grid(samples, n)
    #     self.writer_tb.add_image('samples', grid, epoch)
    #     # log samples on transfer
    #     grid = make_grid(samples_mnist, n)
    #     self.writer_tb.add_image('samples_mnist', grid, epoch)


        #self.writer_wb.log({"conditional samples": [wandb.Image(samples.numpy(), caption="Omniglot")]})
        #self.writer_wb.log({"conditional samples": [wandb.Image(samples_mnist.numpy(), caption="MNIST")]})

        # writer.add_histogram(name, param, epoch) 
        # writer.add_histogram(name, param.grad, epoch)      
