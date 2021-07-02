import threading
import torch
import copy
from torchvision.models import resnet18, vgg16, alexnet
from torch.profiler import profile, record_function, ProfilerActivity


# return a list of layers
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


# move a layer to specific gpu by replacing the forward function
def move_layer(layer, device, job_mutex, scheduler_mutex, i):
    layer.to(device)
    original_forward = layer.forward

    def moved_forward(x):
        print("ready")
        if job_mutex.acquire():
            scheduler_mutex.release()
            x = x.to(device)
            out = original_forward(x)
            job_mutex.release()
            scheduler_mutex.acquire()
            print("forward")
            return out

    layer.forward = moved_forward


# split the model into layers and move them to different gpu
def run_split(net, job_mutex, scheduler_mutex, move=True, batch_size=4, do_profile=True):
    device_count = torch.cuda.device_count()
    scheduler_mutex.acquire()

    if move:
        layers = get_children(net)
        for i, layer in enumerate(layers):
            move_layer(layer, i % device_count, job_mutex, scheduler_mutex, i)
    else:
        net.to(0)

    if do_profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            with record_function("model_inference"):
                inputs = torch.randn(batch_size, 3, 224, 224).to(0)
                net(inputs)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    else:
        inputs = torch.randn(batch_size, 3, 224, 224).to(0)
        net(inputs)


if __name__ == "__main__":
    net = alexnet()
    job_mutex = threading.Lock()
    scheduler_mutex = threading.Lock()
    job_mutex.acquire()
    done = False
    threading.Thread(target=run_split,args=(net, job_mutex, scheduler_mutex, True, 1, True), daemon=True).start()
    while:
        a = input()
        job_mutex.release()
        if scheduler_mutex.acquire():
            job_mutex.acquire()
            scheduler_mutex.release()
