print("HPC works :)")
import os
#os.environ["CUDA_VISIBLE_DEVICES"]= '0' #for jupyter notebook in debugging mode
import time
import json
import torch
import pickle
import random
import datetime
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from collections import defaultdict
from torchvision.io import read_image
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter


import CLIP.clip as clip
from saft import select_saft_parameters
from lr_scheduler import cosine_lr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class dassl_dataset(Dataset):
    def __init__(self, PATH, split="train", transform=None):
        
        with open(PATH, "rb") as file:
            data = pickle.load(file)
            self.train = data[split]
        
        self.transform = transform

    def __len__(self):
        return len(self.train["label"])
    
    def __getitem__(self, idx):
        #img = read_image(path=self.train["impath"][idx], mode="ImageReadMode.RGB")
        img = Image.open(self.train["impath"][idx])

        if self.transform:
            img = self.transform(img)

        return img, self.train["label"][idx]
        
def print_memory_usage():
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")

def print_trainable_params(model):
    count=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            
            count += torch.prod(torch.tensor(param.shape)).item()
            #print(name, param.shape)
    return count

def print_lora_params(model):
    for name, param in model.named_parameters():
        if "lora" in name:
            print(name)

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def get_val_loss(model, val_loader, encoded_classes):
    model.eval()
    val_loss, val_acc, n = 0., 0., 0.
    with torch.no_grad():
        for imgs, labels in val_loader:
            
            imgs = imgs.to(device)
            labels = labels.long().to(device)
            encoded_classes = encoded_classes.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                #logits_per_image, logits_per_text = model(imgs, encoded_classes)
                
                image_features = model.encode_image(imgs)
                text_features = model.encode_text(encoded_classes)

                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ text_features.t()

                probs = logits.softmax(dim=-1)
                #probs = logits_per_image.softmax(dim=-1)
                predicted_class = torch.argmax(probs, dim=1)
                loss = loss_fn(logits, labels)

            val_loss += loss.item()
            acc = torch.sum(predicted_class == labels) 
            val_acc += acc.item()
            n += imgs.size(0)

        return (val_loss / len(val_loader)), (val_acc / n)*100 

#TODO: create config file for hyperparams
#TODO: create trainer function

if __name__ == '__main__':

    torch.manual_seed(0) # reproduceablility
    random.seed(0)

    # hyperparams
    B = 512 # batch size
    lr =  5e-6#1e-5 learning rate
    cons_lr = 1e-5
    wd = 0.1 #  weight decay
    momentum = 0.9
    epochs = 100 
    OG = True # load original model or LoRa model
    saft = True
    r=256 # rank of matrix
    lora_mode="vision+text" # which part of CLIP to lorafy
    val_iter = 1
    print_step = 10
    warmup_steps = 500
    warmup_epochs = 1
    SEED = 2
    comment = "wih fixed temparature of 100"

    set_random_seed(SEED)

    # load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device, OG=OG, r=r, lora_mode=lora_mode, jit=False)
    model.float()

    
    pt_model_dict = deepcopy(model.state_dict())

    # load dataset, create dataloaders
    #train_dataset = torchvision.datasets.ImageNet(root="/fast_storage/mittal/imagenet/images", split="train", transform=preprocess)
    train_dataset = dassl_dataset(PATH=f"/fast_storage/mittal/imagenet/split_fewshot/shot_16-seed_{SEED}.pkl", split="train", transform=preprocess)
    val_dataset = torchvision.datasets.ImageNet(root="/fast_storage/mittal/imagenet/images", split="val", transform=preprocess)
    #val_dataset = dassl_dataset(PATH="/fast_storage/mittal/imagenet/preprocessed.pkl", split="val", transform=preprocess)

    """
    with open ('idx_seed_0', 'rb') as fp:
        sampled_indices = pickle.load(fp)
    train_dataset = Subset(train_dataset, sampled_indices)
    print("Few-shot training prepared!")
    """

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    # read class labels for ImageNet and tokenize them 
    filename = "./imagenet-class_namesOG.json"
    with open(filename) as f:
        imagenet_classes_raw = json.load(f) 
    imagenet_classes = ["a photo of a {}.".format(cl) for cl in imagenet_classes_raw]
    encoded_classes = clip.tokenize(imagenet_classes)
    imagenet_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]

    # other hyperparams
    params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9,0.999), eps=1e-8)
    #optimizer = torch.optim.SGD(
    #        params,
    #        lr=lr,
    #        momentum=momentum,
    #        weight_decay=wd,
    #    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*len(train_loader)) # Learning rate schedule
    #scheduler = cosine_lr(optimizer, lr, warmup_steps, epochs *  len(train_loader))    
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
    log_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = f"{saft}_{SEED}"
    writer = SummaryWriter(log_dir="logs/runs/"+log_tag, comment=comment)

    #zeroshot_weights = zeroshot_classifier(imagenet_classes_raw, imagenet_templates)
    params = print_trainable_params(model)
    print(f"Number of trainable params: {params/10**6:.2f}M")
    print(f"Total number of steps in 1 training epoch: {len(train_dataset)/B:.1f}")
    print(f"Total number of steps in validation set: {len(val_dataset)/(B*2):.1f}")
    print("\n")
    
    if saft:
        #saft_mask, _, _, _, _ = select_saft_parameters(model, train_loader, loss_fn, device, encoded_classes, 0.001)
        with open(f'saft_mask_{SEED}.pkl', 'rb') as f:
            saft_mask = pickle.load(f)
        print("Generated saft mask!")
        #model, _ = clip.load("ViT-B/16", device=device, OG=OG, r=r, lora_mode=lora_mode, jit=False)
        #model.float()

    prof.start()
    iter_num, global_val_loss = 0, 0

    # training loop
    for epoch in range(epochs):
        
        
            
            # perform validation
            
            
        if iter_num % val_iter == 0:
            
            val_loss, val_acc = get_val_loss(model, val_loader, encoded_classes)
            print(f"----------- val loss {val_loss:.4f} | val accuracy {val_acc:.2f}% -----------")
            writer.add_scalar("Loss/val", val_loss, iter_num)
            writer.add_scalar("Acc/val", val_acc, iter_num)
            
            if iter_num != 0:
                if global_val_loss > val_loss :
                    global_val_loss = val_loss
                    checkpoint = {"model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                #"scaler": scaler.state_dict(),
                                "scheduler": scheduler.state_dict()
                                }
                    
                    # Write checkpoint
                    print("Saving the new checkpoint... :)")
                    torch.save(checkpoint, f"bestCLIP_{model_tag}.pt")
            else:
                global_val_loss = val_loss
        
        for i, (imgs, labels) in enumerate(train_loader):
            
            model.train()
            prof.step()
            t0 = time.time()
            step = i + epoch * B
            #scheduler(step)

            imgs = imgs.to(device)
            labels = labels.long().to(device)
            encoded_classes = encoded_classes.to(device)
            optimizer.zero_grad()

            # forward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                #logits_per_image, logits_per_text = model(imgs, encoded_classes)
                #probs = logits_per_image.softmax(dim=-1)
                
                image_features = model.encode_image(imgs)
                text_features = model.encode_text(encoded_classes)

                # normalized features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ text_features.t()

                probs = logits.softmax(dim=-1)
                predicted_class = torch.argmax(probs, dim=1)
                loss = loss_fn(logits, labels)

            # calculate the accuracy
            train_acc = torch.sum(predicted_class == labels) / labels.size(0)

            # backprop with grad sacling and clipping
            #loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping above norm 1.0
            scaler.step(optimizer)
            #optimizer.step()
            scheduler.step()
            scaler.update()
            

            if saft:
                current_model_dict = {}
                #print(current_model_dict["visual.transformer.resblocks.11.mlp.c_proj.weight"])
                for key,value in model.state_dict().items():
                    try:
                        current_model_dict[key] = value*saft_mask[key] + pt_model_dict[key]*~saft_mask[key]
                    except:
                        print(key)
                
                model.load_state_dict(current_model_dict)

                #print(saft_mask["visual.transformer.resblocks.11.mlp.c_proj.weight"])
                #print(pt_model_dict["visual.transformer.resblocks.11.mlp.c_proj.weight"])
                #print(current_model_dict["visual.transformer.resblocks.11.mlp.c_proj.weight"])

            #for param, val in model.named_parameters():
            #    print(param)
            #    if param == "visual.transformer.resblocks.11.mlp.c_proj.weight":
            #       print(val.grad)

            #inp = input("Test....")

            torch.cuda.synchronize()
            t1 = time.time()
            dt = (t1-t0)*1000 # time difference in mili seconds
            iter_num += 1
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("Loss/train", loss, iter_num)
            writer.add_scalar("Acc/train", train_acc, iter_num)
            writer.add_scalar("lr", current_lr, iter_num)

            if iter_num % print_step == 0:
                print(f"step/epoch {iter_num}/{epoch}: train loss {loss.item():.4f} | time {dt:.2f}ms | lr {current_lr} | accuracy {100*train_acc.detach().item():.2f}")
                #print(f"step/epoch {iter_num}/{epoch}: train loss {loss.item():.4f} | time {dt:.2f}ms | accuracy {100*train_acc.detach().item():.2f}")

        #print(f"-----------best validation accuray till epoch {epoch}: {global_val_acc}%-----------")
        
        

            
    #TODO:  add args 
    prof.stop()
    writer.flush()
