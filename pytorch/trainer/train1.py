import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from loss.loss import dice_coef
import datetime
from utils.util import save_best

def validation(epoch, model, data_loader, criterion, classes, thr=0.5):
    print(f'검증 시작 epoch : {epoch}')
    model.eval()
    model = model.cuda()
    n_class = len(classes)
    
    dices = []
    with torch.no_grad():    
        total_loss = 0
        cnt = 0
        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w))
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(masks, outputs)
            dices.append(dice)
            
            torch.cuda.empty_cache()
            
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(classes, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    return avg_dice

def train(model, num_epoch, data_loader, val_step, val_loader, criterion, optimizer, save_dir, classes):
    n_class = len(classes)
    best_dice = 0
    model.cuda()
    
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        pbar = tqdm(data_loader, desc=f"epoch [{epoch+1}/{num_epoch}]")
        for step, (images, masks) in enumerate(pbar):
            
            
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix(loss=round(loss.item(), 4))
            
        if (epoch + 1) % val_step == 0:
            dice = validation(epoch +1, model, val_loader, criterion, classes)
            
            if best_dice < dice:
                print(f"Best epoch : epoch {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                best_dice = dice
                save_best(model, save_dir)