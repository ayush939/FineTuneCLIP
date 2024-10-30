print("HPC works :)")


import os
#os.environ["CUDA_VISIBLE_DEVICES"]= '0'
import time
import json
import torch
import datetime
import torchvision
import numpy as np
from tqdm import tqdm
from torch.profiler import ProfilerActivity
from torch.utils.tensorboard import SummaryWriter


import CLIP.clip as clip

torch.manual_seed(0) # reproduceablility

# hyperparams
B = 512 # batch size
lr = 3e-5 # learning rate
wd = 1e-1 #  weight decay
epochs = 10 
OG = True # load original model or LoRa model
r=256 # rank of matrix
lora_mode="vision+text" # which part of CLIP to lorafy
val_iter = 1000
print_step = 100
warmup_steps = 500
comment = "wih fix temparature of 100"

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, OG=OG, r=r, lora_mode=lora_mode, jit=False)
model.float()

# load dataset, create dataloaders
train_dataset = torchvision.datasets.ImageNet(root="/fast_storage/mittal/", split="train", transform=preprocess)
val_dataset = torchvision.datasets.ImageNet(root="/fast_storage/mittal/", split="val", transform=preprocess)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=B*2, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

# read class labels for ImageNet and tokenize them 
filename = "./imagenet-class_namesOG.json"
with open(filename) as f:
    imagenet_classes_raw = json.load(f) 
imagenet_classes = ["a photo of a {}".format(cl) for cl in imagenet_classes_raw]
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
cosineAnnealingLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, warmup_steps) # Learning rate schedule
scaler = torch.amp.GradScaler('cuda', enabled=True)
#scheduler = cosine_lr(optimizer, lr, warmup_length, epochs * num_batches)

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
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                logits = 100. * image_features @ text_features.t()

                probs = logits.softmax(dim=-1)
                #probs = logits_per_image.softmax(dim=-1)
                predicted_class = torch.argmax(probs, dim=1)
                loss = loss_fn(logits, labels)

            val_loss += loss.item()
            acc = torch.sum(predicted_class == labels) 
            val_acc += acc.item()
            n += imgs.size(0)

        return (val_loss / n), (val_acc / n)*100 

prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
log_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir="logs/runs/"+log_tag, comment=comment)

#zeroshot_weights = zeroshot_classifier(imagenet_classes_raw, imagenet_templates)
params = print_trainable_params(model)
print(f"Number of trainable params: {params/10**6:.2f}M")
print(f"Total number of steps in 1 training epoch: {len(train_dataset)/B:.1f}")
print(f"Total number of steps in validation set: {len(val_dataset)/(B*2):.1f}")
print("\n")

prof.start()
iter_num, global_val_acc = 0, 0

# training loop
for epoch in range(epochs):
    
    for imgs, labels in train_loader:
        
        # perform validation
        if iter_num % val_iter == 0:
            
            val_loss, val_acc = get_val_loss(model, val_loader, encoded_classes)
            print(f"----------- val loss {val_loss:.4f} | val accuracy {val_acc:.2f}% -----------")
            writer.add_scalar("Loss/val", val_loss, iter_num)
            writer.add_scalar("Acc/val", val_acc, iter_num)
            
            if iter_num != 0:
                if global_val_acc < val_acc :
                    global_val_acc = val_acc
                    checkpoint = {"model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(),
                                "scheduler": cosineAnnealingLR.state_dict()}
                    
                    # Write checkpoint
                    print("Saving the new checkpoint... :)")
                    torch.save(checkpoint, f"bestCLIP_{log_tag}.pt")
            else:
                global_val_acc = val_acc

        
        model.train()
        prof.step()
        t0 = time.time()

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
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            logits = 100. * image_features @ text_features.t()

            probs = logits.softmax(dim=-1)
            predicted_class = torch.argmax(probs, dim=1)
            loss = loss_fn(logits, labels)

        # calculate the accuracy
        train_acc = torch.sum(predicted_class == labels) / labels.size(0)

        # backprop with grad sacling and clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping above norm 1.0
        scaler.step(optimizer)
        cosineAnnealingLR.step()
        scaler.update()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000 # time difference in mili seconds
        iter_num += 1
        current_lr = cosineAnnealingLR.get_last_lr()[0]
        writer.add_scalar("Loss/train", loss, iter_num)
        writer.add_scalar("Acc/train", train_acc, iter_num)
        writer.add_scalar("lr", current_lr, iter_num)

        if iter_num % print_step == 0:
            print(f"step/epoch {iter_num}/{epoch}: train loss {loss.item():.4f} | time {dt:.2f}ms | lr {current_lr} | accuracy {100*train_acc.detach().item():.2f}")

    print(f"-----------best validation accuray till epoch {epoch}: {global_val_acc}%-----------")
        
#TODO:  add args 
prof.stop()
writer.flush()
