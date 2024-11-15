import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from trainer.test_rle import encode_mask_to_rle
from utils.util import get_IND2CLASS

def test(args, data_loader, thr=0.5):
    model = torch.load(args.model)
    model = model.cuda()
    model.eval()
    rles = []
    filename_and_class = []
    with torch.no_grad():
        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{get_IND2CLASS()[c]}_{image_name}")
    return rles, filename_and_class