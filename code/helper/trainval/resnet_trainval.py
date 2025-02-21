from helper.models.config import Config
from torch import argmax, no_grad, set_grad_enabled, enable_grad, long, Tensor, round, unsqueeze
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import torch.nn.functional as F
from helper.trainval.metrics import get_iou, get_acc, get_prec, get_recall, get_dice, get_f1
from torchvision import transforms
from helper.callbacks.visualize import show_prediction
from numpy import transpose, float32


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.model.train()
    
    running_loss = 0.0
    running_corrects = 0.0
    total_batches = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        print(f"images: {images.shape}")
        print(f"masks: {masks.shape}")
        
        outputs = model.model(images)["out"]
        print(f"output model is {outputs.shape}")
        #print(f"argmaxed outputs = {outputs}, shape={outputs.size()}")
        
        loss = criterion(outputs, masks.squeeze())

        outputs = argmax(outputs, dim=1)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item() * images.size(0)
        # print(f"train predicted {outputs} shape:{outputs.shape}")
        # print(f"masks train{masks.squeeze()}, shape:{masks.squeeze().shape}")
        
        running_corrects += get_iou(outputs, masks.squeeze())
        
        total_batches += 1


    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / total_batches
    
    return epoch_loss, epoch_acc.cpu().item()
    
    
def val_epoch(model, dataloader, criterion, optimizer, device):
    
    model.model.eval()
    val_loss = 0.0
    val_corrects = 0.0
    total_batches = 0
    
    with no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model.model(images)['out']
            loss = criterion(outputs, masks.squeeze())
            val_loss += loss.item() * images.size(0)
            
            predicted = argmax(outputs, dim=1)
            #print(f"Val predicted {predicted}, masks val{masks.squeeze()}")
            val_corrects += get_iou(predicted, masks.squeeze())
            
            total_batches += 1
    
    val_loss /= len(dataloader.dataset)
    val_acc = val_corrects / total_batches
    
    return val_loss, val_acc.cpu().item()
    

def train_model(model, dataloaders, config: Config, device):
    print(f"Training model {model.get_name()} - {model.counter} using {device}")
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # log_template = "\nEpoch {ep:02d} train_loss: {t_loss} \
    #     val_{} {v_loss:0.4f} train_{} {t_acc:0.4f} val_acc {v_acc:0.4f}"
    plt.ioff()
    
    fig, ax = plt.subplots(1, 1)
    hdisplay = display.display('', display_id=True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    
    train_loss_line, = ax.plot([], [], label='Train Loss')
    val_loss_line, = ax.plot([], [], label='Validation Loss')
    ax.legend()
    
    with tqdm(desc="epoch", total=config.NUM_EPOCHS) as pbar_outer:
        optimizer = config.optimizer
        criterion = config.criterion
        
        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
            
            val_loss, val_acc = val_epoch(model, dataloaders['validate'], criterion, optimizer, device)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            pbar_outer.update(1)
            print(f"Epoch {epoch}: train_loss {train_loss}, train_iou {train_acc},  val_loss {val_loss}, val_iou {val_acc}")
            
            train_loss_line.set_data(range(1, epoch + 2), history['train_loss'])
            val_loss_line.set_data(range(1, epoch + 2), history['val_loss'])
            
            ax.set_xlim(0, config.NUM_EPOCHS + 2)
            ax.set_ylim(0, max(max(history['train_loss']), max(history['val_loss'])) + 1)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            hdisplay.update(fig)
            
    return history


def predict(model, image, mask, device):
    model.eval()  # Модель в режиме оценки

    # Преобразуем изображения в тензоры и отправим на нужное устройство
    image = image.to(device)
    mask = mask.to(device)
    image = unsqueeze(image, 0)
    mask = unsqueeze(mask, 0)
    with no_grad():        
            
        output = model(pixel_values=image).logits
        output = F.interpolate(output, size=(mask.shape[-2:]), mode='bilinear', align_corners=False)
        pred = argmax(output, dim=1).squeeze(0).cpu()  # Получаем предсказание
            
        show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu())


def test_epoch(model, dataloader, criterion, device, detailed=False):
    model.model.to(device)
    test_loss = 0.0
    total_batches = 0
    
    test_iou = 0.0
    test_acc = 0.0
    test_prec = 0.0
    test_recall = 0.0
    test_f1 = 0.0
    test_dice = 0.0
    
    with no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model.model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=(masks.shape[-2:]), mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks.squeeze())
            test_loss += loss.item() * images.size(0)
            
            predicted = argmax(outputs, dim=1)
            test_iou += get_iou(predicted, masks.squeeze())
            if detailed:
                test_acc += get_acc(predicted, masks.squeeze())
                test_prec += get_prec(predicted, masks.squeeze())
                test_recall += get_recall(predicted, masks.squeeze())
                test_f1 += get_f1(predicted, masks.squeeze())
                test_dice += get_dice(predicted, masks.squeeze())
            
            total_batches += 1
    
    test_loss /= len(dataloader.dataset)
    test_iou = (test_iou / total_batches)
    
    if detailed:
        test_acc = test_acc / total_batches
        test_prec = test_prec / total_batches
        test_recall = test_recall / total_batches
        test_f1 = test_f1 / total_batches
        test_dice = test_dice / total_batches

        return  {'loss': test_loss, 'acc': test_acc.cpu().numpy().item(), 'prec': test_prec.cpu().numpy().item(), 'recall': test_recall.cpu().numpy().item(), 'f1': test_f1.cpu().numpy().item(), 'dice': test_dice.cpu().numpy().item(), 'iou': test_iou.cpu().numpy().item()}
    
    return {'loss': test_loss, 'iou': test_iou.cpu()}
    