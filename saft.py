from copy import deepcopy
import json
import torch
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision

import CLIP.clip as clip


def get_clip_forward_pass(model, imgs, encoded_classes):
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
            
    return logits

def select_saft_parameters(model, dataloader, criterion, device, encoded_classes=None, sparsity_level=0.001):

    """
    Accumulate gradients and select saft based parameters based on gradient magnitudes.

    Args:
        model (torch.nn.Module): The pre-trained model.
        dataloader (DataLoader): DataLoader for the downstream task.
        criterion (callable): Loss function (e.g., CrossEntropyLoss).
        sparsity_level (float): Fraction of parameters to select for fine-tuning.

    Returns:
        dict: Dictionary of selected parameters (names and values).
        int: Total number of parameters.
        int: Total number of selected parameters.
    """
    model.train()  # Ensure model is in evaluation mode
    model.zero_grad()
    
    #param_dict = deepcopy(model.state_dict())
    param_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    total_samples = 0
    

    # Iterate over batches without zeroing gradients
    for inputs, targets in dataloader:
        
        # You can uncomment following line if you have same device error
        #inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        #outputs = model(inputs)

        outputs = get_clip_forward_pass(model, inputs, encoded_classes.to(device))
        # You can uncomment following to handle case where model returns a tuple
        #if isinstance(outputs, tuple):
        #    outputs = outputs[0]

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass to compute gradients
        loss.backward()

        # Accumulate gradients for all parameters
        for name, param in model.named_parameters():
            if param.grad is not None:  # Only accumulate if parameter has gradients
                param_gradients[name] += param.grad
            
        total_samples += len(targets)

        model.zero_grad()

    # Average gradients across all samples
    for name in param_gradients:
        param_gradients[name] /= total_samples
        param_gradients[name] = param_gradients[name].abs()

    # Flatten and sort all gradients to determine the threshold
    all_gradients = torch.cat([g.view(-1) for g in param_gradients.values()])

    # Sort all gradients in descending order
    sorted_gradients, _ = torch.sort(all_gradients, descending=True)

    # Determine the number of parameters to select
    num_selected = int(len(sorted_gradients) * sparsity_level)
    if num_selected < 1: # to prevent zero error
        num_selected = 1 
    threshold = sorted_gradients[num_selected - 1]  # Threshold value for top gradients

    # Select important parameters based on the threshold
    selected_parameters = {}
    total_selected = 0  # Count total selected parameters
    for name, grad in param_gradients.items():
        #if torch.any(grad >= threshold):
        mask = grad >= threshold  # Create a mask for selected values
        selected_parameters[name] = mask
        total_selected += mask.sum().item()

    total_parameters = sum(g.numel() for g in param_gradients.values())
    return selected_parameters, total_parameters, total_selected, threshold, param_gradients


if __name__ == '__main__':


    from clip_imagenet import dassl_dataset

    B = 512
    SEED = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device, OG=True, r=256, lora_mode="text+image", jit=False)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    train_dataset = dassl_dataset(PATH=f"/fast_storage/mittal/imagenet/split_fewshot/shot_16-seed_{SEED}.pkl", split="train", transform=preprocess)

    print("Few-shot training data prepared!")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2)

    # read class labels for ImageNet and tokenize them 
    filename = "./imagenet-class_namesOG.json"
    with open(filename) as f:
        imagenet_classes_raw = json.load(f) 
    imagenet_classes = ["a photo of a {}.".format(cl) for cl in imagenet_classes_raw]
    encoded_classes = clip.tokenize(imagenet_classes)


    saft_state_dict, _, _, _, _ = select_saft_parameters(model, train_loader, loss_fn, device, encoded_classes)

    import pickle 

    with open(f'saft_mask_{SEED}.pkl', 'wb') as f:
        pickle.dump(saft_state_dict, f)
            
    

     

