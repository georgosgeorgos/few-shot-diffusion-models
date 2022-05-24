
from model.vfsddpm import DDPM
from model.vfsddpm import VFSDDPM

def select_model(args):
    if args.model == "ddpm":
        return DDPM
    elif args.model == "vfsddpm":
        return VFSDDPM
    else:
        print("No valid model selected. Please choose {ddpm, vfsddpm}")