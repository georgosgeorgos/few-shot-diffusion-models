import torch
import os
from utils.util import process_batch
from torchvision.utils import save_image


def save_test_grid(inputs, samples, save_path, n=10):
    inputs = 1 - inputs.cpu().data.view(-1, 5, 1, 28, 28)[:n]
    reconstructions = samples.cpu().data.view(-1, 5, 1, 28, 28)[:n]
    images = torch.cat((inputs, reconstructions), dim=1).view(-1, 1, 28, 28)
    save_image(images, save_path, nrow=n)
    return images


def vis_conditional_samples(args, 
                            epoch, 
                            model, 
                            omni_test_batch, 
                            mnist_test_batch
                            ):
    # unseen Omniglot
    filename = args.name + "_" + args.timestamp + "_grid-omniglot_{}.png".format(epoch + 1)
    path = os.path.join(args.fig_dir, filename)
    with torch.no_grad():
        x = omni_test_batch.to(args.device)
        if args.model == "hfsgm":
            samples = model.conditional_sample_cq(x)["xp"]
        else:
            samples = model.conditional_sample_cqL(x)["xp"]
    conditional_samples = save_test_grid(x, samples, path)
    
    # unseen MNIST
    filename = args.name + "_" + args.timestamp + "_grid-mnist_{}.png".format(epoch + 1)
    path = os.path.join(args.fig_dir, filename)
    with torch.no_grad():
        x = mnist_test_batch.to(args.device)
        if args.model in ["hfsgm", "thfsgm"]:
            samples_mnist = model.conditional_sample_cq(x)["xp"]
        else:
            samples_mnist = model.conditional_sample_cqL(x)["xp"]
    conditional_samples_mnist = save_test_grid(x, samples_mnist, path)
    return conditional_samples, conditional_samples_mnist



def vis_batch():
    return None