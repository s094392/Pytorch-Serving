import torch
import copy
from torchvision.models import resnet18, vgg16, alexnet
from torch.profiler import profile, record_function, ProfilerActivity

def get_children(model: torch.nn.Module):
    # get children form model
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child
        return model
    else:
       # look for children from children... to the last child
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def move_layer(layer, device):
    layer.to(device)
    original_forward = layer.forward

    def moved_forward(x):
        x = x.to(device).float()
        return original_forward(x)
        
    layer.forward = moved_forward

if __name__ == "__main__":
    net = alexnet()
    layers = get_children(net)
    for i in range(len(layers)):
        move_layer(layers[i], f"cuda:{i%1}")
        
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            batch_size = 4
            inputs = torch.randn(batch_size, 3, 224, 224)
            net(inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total"))
